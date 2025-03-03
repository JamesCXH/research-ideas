
class VisionEncoder(nn.Module):
    def __init__(
        self,
        img_size=64,
        patch_size=16,
        in_chans=3,
        embed_dim=256,
        depth=6,
        num_heads=1,
    ):
        super(VisionEncoder, self).__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size."
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.grid_size = img_size // patch_size  # Number of patches along one dimension

        # Patch embedding layer
        self.patch_embed = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # **Replaced**: Positional embeddings with fixed 2D sine-cosine embeddings
        pos_embed = self._get_2d_sincos_pos_embed(embed_dim, self.grid_size)
        pos_embed = pos_embed.unsqueeze(0)  # Shape: (1, num_patches, embed_dim)
        self.register_buffer('pos_embed', pos_embed)  # Not a parameter

        # Transformer encoder layers as a ModuleList
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads, batch_first=True,
                dim_feedforward=embed_dim * 4, dropout=0
            ) for _ in range(depth)
        ])

    def patch_and_embed(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # Shape: (batch_size, embed_dim, H_p, W_p)
        x = x.flatten(2)         # Shape: (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)    # Shape: (batch_size, num_patches, embed_dim)

        # Add fixed 2D sine-cosine positional embeddings
        x = x + self.pos_embed   # Shape: (batch_size, num_patches, embed_dim)

        return x

    def run_transformer(self, x):
        # Collect outputs from all layers
        layer_outputs = []
        for layer in self.encoder_layers:
            x = layer(x)  # Shape: (batch_size, num_patches, embed_dim)
            layer_outputs.append(x.unsqueeze(1))  # Shape: (batch_size, 1, num_patches, embed_dim)

        # Stack all layer outputs: Shape (batch_size, depth, num_patches, embed_dim)
        stacked_outputs = torch.cat(layer_outputs, dim=1)

        return stacked_outputs


    def forward(self, x):
        """
        x: Input tensor of shape (batch_size, 3, img_size, img_size)
        Returns:
            weighted_output: Tensor of shape (batch_size, num_patches, embed_dim)
        """
        batch_size = x.size(0)

        x = self.patch_and_embed(x)

        stacked_outputs = self.run_transformer(x)

        return stacked_outputs

    def _get_2d_sincos_pos_embed(self, embed_dim, grid_size):
        """
        Generate 2D sine-cosine positional embeddings.

        Args:
            embed_dim (int): Dimension of the embeddings. Must be divisible by 2 * grid_size.
            grid_size (int): Number of patches along one dimension.

        Returns:
            torch.Tensor: Positional embeddings of shape (num_patches, embed_dim)
        """
        assert embed_dim % 2 == 0, "Embed dimension must be even for sine and cosine."

        # Compute the number of patches per axis
        grid_h = torch.arange(grid_size, dtype=torch.float32)
        grid_w = torch.arange(grid_size, dtype=torch.float32)
        grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing='ij')  # Shape: (grid_size, grid_size)

        grid_h = grid_h.flatten()  # Shape: (num_patches,)
        grid_w = grid_w.flatten()  # Shape: (num_patches,)

        # Compute sine and cosine embeddings for height and width
        pos_h = self._get_1d_sincos_pos_embed(embed_dim // 2, grid_h)  # Shape: (num_patches, embed_dim // 2)
        pos_w = self._get_1d_sincos_pos_embed(embed_dim // 2, grid_w)  # Shape: (num_patches, embed_dim // 2)

        # Concatenate height and width embeddings
        pos = torch.cat([pos_h, pos_w], dim=1)  # Shape: (num_patches, embed_dim)

        return pos

    def _get_1d_sincos_pos_embed(self, embed_dim, pos):
        """
        Generate 1D sine-cosine positional embeddings.

        Args:
            embed_dim (int): Dimension of the embeddings. Must be even.
            pos (torch.Tensor): Positions (integer indices).

        Returns:
            torch.Tensor: Positional embeddings of shape (num_patches, embed_dim)
        """
        assert embed_dim % 2 == 0, "Embed dimension must be even for sine and cosine."

        omega = torch.arange(embed_dim // 2, dtype=torch.float32) / (embed_dim / 2)
        omega = 1. / (10000 ** omega)  # Shape: (embed_dim // 2,)

        pos = pos[:, None] * omega[None, :]  # Shape: (num_patches, embed_dim // 2)
        emb_sin = torch.sin(pos)             # Shape: (num_patches, embed_dim // 2)
        emb_cos = torch.cos(pos)             # Shape: (num_patches, embed_dim // 2)
        emb = torch.cat([emb_sin, emb_cos], dim=1)  # Shape: (num_patches, embed_dim)

        return emb

class PairedBYOLAugment(nn.Module):
    """
    Applies BYOL-style image augmentations to paired inputs.

    Each sample i in the batch receives the same random transformation for both views (x1[i] and x2[i]).

    BYOL augmentations include:
      • Random Resized Crop (applied only if `crop` is True): A random crop with an area uniformly sampled between 8% and 100% of the original image,
        and an aspect ratio sampled between 3/4 and 4/3, resized to 64×64 using bicubic interpolation.
      • Random Horizontal Flip: Flips the image left-right with a probability of 0.5.
      • Color Jitter: Randomly adjusts brightness, contrast, saturation, and hue (brightness=0.4, contrast=0.4,
        saturation=0.4, hue=0.1) with a probability of 0.8.
      • Random Grayscale: Converts the image to grayscale with a probability of 0.2.
      • Random Gaussian Blur: Applies Gaussian blur with a kernel size of 7×7 and a sigma uniformly sampled
        from [0.1, 2.0].
      • Random Solarization: Applies solarization with a threshold of 0.5 and a probability of 0.2.
      • (Optional) Random Erasing: Randomly erases a part of the image with a probability of 0.5.

    The same randomly generated parameters are applied to both inputs.
    """
    def __init__(self):
        super().__init__()

        # Always applied augmentations.
        transforms = [
            # Optional left-right flip.
            K.RandomHorizontalFlip(p=0.5),
            # Color jittering with random order of brightness, contrast, saturation, and hue adjustments.
            K.RandomResizedCrop(
                size=(64, 64),
                scale=(0.7, 1.0),
                ratio=(0.9, 1.1),
                p=0.5
            ),

            K.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1,
                p=0.8
            ),
            # Random grayscale conversion.
            K.RandomGrayscale(p=0.2),
            # Gaussian blur.
            K.RandomGaussianBlur(
                kernel_size=(7, 7),
                sigma=(0.1, 2.0),
                p=1.0
            ),
            # Solarization.
            K.RandomSolarize(
                thresholds=(0.5, 0.5),
                p=0.2
            )
        ]

        # Ensure the same augmentation parameters are applied to both images in a pair.
        self.transforms = K.AugmentationSequential(*transforms)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        Args:
            x1, x2 (torch.Tensor): Input images in the range [0, 1] with shape [B, C, H, W].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The augmented images with the same random transform applied per sample.
        """
        # Generate random parameters for the batch.
        params = self.transforms.forward_parameters(x1.shape)
        # Apply the same transformation parameters to both inputs.
        x1_out = self.transforms(x1, params=params)
        x2_out = self.transforms(x2, params=params)


        return x1_out, x2_out

class DoubleCrossDecoderLayer(nn.Module):
    """
    A Transformer-style decoder layer that attends to memory_action with a pad mask.
    """
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        # For simplicity, we only keep the relevant cross-attn to actions here
        self.cross_attn_action = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, memory_state, memory_action, action_mask=None):
        """
        memory_state : [B, T_state, d_model]
        memory_action: [B, T_action, d_model]
        action_mask  : [B, T_action] with True for padded positions (or None)
        """
        # cross-attend to action
        x, _ = self.cross_attn_action(
            memory_state,
            memory_action,
            memory_action,
            key_padding_mask=action_mask
        )
        memory_state = self.norm3(memory_state + x)

        x = self.linear2(self.dropout(self.activation(self.linear1(memory_state))))
        memory_state = self.norm4(memory_state + x)

        return memory_state


class DoubleCrossDecoder(nn.Module):
    """
    A stack of DoubleCrossDecoderLayer that optionally takes action_mask.
    """
    def __init__(self, d_model, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DoubleCrossDecoderLayer(d_model, nhead=nhead, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, memory_state, memory_action, action_mask=None):
        """
        memory_state: [B, T_state, d_model]
        memory_action: [B, T_action, d_model]
        action_mask: [B, T_action] or None
        """
        out = memory_state
        for layer in self.layers:
            out = layer(out, memory_action, action_mask=action_mask)
        return out


class WorldModel(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        action_dim=128,
        img_size=64,
        patch_size=8,
        wm_num_heads=1,
        wm_depth=1,
        ve_depth=3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.ve_depth = ve_depth

        # Assume you have a VisionEncoder + EMA
        self.encoder = VisionEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=ve_depth,
            num_heads=4,
        )
        self.ema_encoder = copy.deepcopy(self.encoder)
        for p in self.ema_encoder.parameters():
            p.requires_grad_(False)

        self.num_patches = self.encoder.num_patches
        self.d_model = self.ve_depth * self.embed_dim
        # self.d_model = self.embed_dim

        # Action encoder: from action_dim => d_model
        self.action_encoder = nn.Linear(self.action_dim, self.d_model)

        # DoubleCrossDecoder that can accept action_mask
        self.decoder = DoubleCrossDecoder(
            d_model=self.d_model,
            nhead=ve_depth,
            num_layers=wm_depth,
            dropout=0.0,
        )

    def forward(self, before_obs: torch.Tensor, action: torch.Tensor, action_mask=None):
        """
        before_obs: [B, C, H, W]
        action:     [B, T, action_dim] (padded in T dimension if needed)
        action_mask:[B, T] = True for padded positions, or None

        Returns predicted next-state embedding: [B, D, P, E] shape if you choose,
        or a simpler shape depending on your usage.
        """
        B = before_obs.shape[0]

        # Encode "before_obs" with EMA
        with torch.no_grad():
            # shape [B, depth, P, E]
            state_embed = self.ema_encoder(before_obs)
            # global center
            # global_center = state_embed.mean()
            # state_embed = state_embed - global_center

        # Flatten [B, P, depth*E]
        P = state_embed.shape[2]
        D = state_embed.shape[1]
        E = state_embed.shape[3]
        state_embed = state_embed.permute(0,2,1,3).reshape(B, P, D*E)  # => [B, P, d_model]

        # Encode action => memory_action [B, T, d_model]
        memory_action = self.action_encoder(action)

        # Pass through the DoubleCrossDecoder
        out = self.decoder(
            memory_state=state_embed,
            memory_action=memory_action,
            action_mask=action_mask
        )
        # out => [B, P, d_model]
        # Reshape back => [B, D, P, E]
        out = out.view(B, P, D, E).permute(0, 2, 1, 3)
        return out

    def label(self, batch):
        """
        Same logic as your original code: next-state L2 to EMA next_obs.
        We'll also retrieve la_mask if present to avoid padded tokens in cross-attn.
        """
        la = batch["la"]  # shape [B, T, action_dim]
        la_mask = batch.get("la_mask", None)  # shape [B, T] if it exists
        before_obs = batch["obs"][:, -2]
        target_obs = batch["obs"][:, -1]

        # Next-state predictions
        pred = self(before_obs, la, action_mask=la_mask)

        # Meanwhile we get the target embedding
        with torch.no_grad():
            target_embed = self.ema_encoder(target_obs)
            # target_embed = target_embed - target_embed.mean()

        diff = F.mse_loss(pred, target_embed, reduction='none')
        loss_per_depth = diff.mean(dim=(0,2,3))  # e.g. [D]
        return loss_per_depth

    @torch.no_grad()
    def update_ema(self, momentum=0.9):
        """
        EMA update for the online encoder -> target/EMA encoder.
        """
        for ema_param, param in zip(self.ema_encoder.parameters(), self.encoder.parameters()):
            ema_param.data.mul_(momentum).add_(param.data * (1 - momentum))

    def get_encoder(self):
        return self.encoder

    def get_ema_encoder(self):
        return self.ema_encoder

def generate_causal_mask(L: int, device) -> torch.Tensor:
    """
    Returns [L, L] bool mask, where mask[i,j] = True
    if position i cannot attend to position j (j > i).
    """
    mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()
    return mask


class IDM(nn.Module):
    """
    Inverse Dynamics Model (IDM) updated so that:
      - For the "after" patches during training, rather than dropping them out,
        we run all patches through the transformer and then (using a computed keep-mask)
        replace the positions that were meant to be dropped with a learned drop token.
      - The before and after patch features are kept in full (without time-step embeddings)
        and concatenated along the last dimension, so the resulting context has dimension:
            depth * embed_dim * 2.
      - The Transformer (and subsequent tokens) is updated accordingly.
    """
    def __init__(
        self,
        action_dim: int,
        encoder: VisionEncoder,      # main (trainable) vision encoder
        ema_encoder: VisionEncoder,  # optional EMA copy
        top_pct=0.1,
        min_tokens=2,
        max_tokens=10,
        num_eval_tokens=10,
        nhead=8,
        num_encoder_layers=0,
        num_decoder_layers=1,
    ):
        super().__init__()
        self.encoder = encoder
        self.ema_encoder = ema_encoder
        self.ema_encoder.requires_grad_(False)

        # Vision encoder output dimensions
        self.depth = self.encoder.depth
        self.embed_dim = self.encoder.embed_dim
        self.num_patches = self.encoder.num_patches
        # Previously: self.d_context = depth * embed_dim; now we double it via concatenation
        self.d_context = self.depth * self.embed_dim * 2

        # AR token counts
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.num_eval_tokens = num_eval_tokens
        self.top_pct = top_pct

        # Data augmentation
        self.aug_fn = PairedBYOLAugment()

        self.transformer = nn.Transformer(
            d_model=self.d_context,
            dim_feedforward=self.d_context * 4,
            nhead=self.depth,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            batch_first=True,   # shapes are [B, S, E]
        )

        # Learned "mask" token for AR decode (shape [1, 1, d_context])
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.d_context) * 0.02)

        # Learned drop token to use for "dropped" after patches (each of dim depth*embed_dim)
        self.drop_token = nn.Parameter(torch.randn(1, 1, self.depth * self.embed_dim) * 0.02)

        # Linear head to mix/adjust the combined vision context (after concatenation)
        # self.embed_head = nn.Linear(self.d_context, self.d_context)

        # Final action head
        self.action_head = nn.Linear(self.d_context, action_dim)

    def _make_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """
        Returns a [size, size] float mask for causal decoding,
        where mask[i, j] = -inf if j > i (no attending to future tokens).
        """
        mask = torch.full((size, size), float('-inf'), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def _encode_vision_context(self, x_before, x_after, training=True):
        device = x_before.device
        B = x_before.shape[0]

        if training:
            with torch.no_grad():
                x_b_aug, x_a_aug = self.aug_fn(x_before, x_after)
        else:
            x_b_aug, x_a_aug = x_before, x_after

        # Patch+embed for both images: [B, P, embed_dim]
        before_patch = self.encoder.patch_and_embed(x_b_aug)
        after_patch  = self.encoder.patch_and_embed(x_a_aug)

        P = after_patch.shape[1]
        if training:
            top_k = max(1, int(P * self.top_pct))
            rand_vals = torch.rand(B, P, device=device)
            _, idxs = rand_vals.topk(top_k, largest=False)
            keep_mask = torch.zeros(B, P, dtype=torch.bool, device=device)
            batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, top_k)
            keep_mask[batch_idx, idxs] = True
        else:
            idxs = torch.arange(P, device=device).unsqueeze(0).expand(B, -1)  # [B, P]

        # --- Process the before branch ---
        before_enc = self.encoder.run_transformer(before_patch)  # [B, D, P, embed_dim]
        # Rearrange to [B, P, D, embed_dim] without flattening the depth:
        before_ctx = before_enc.transpose(1, 2).reshape(B, P, self.depth, self.embed_dim)

        # --- Process the after branch ---
        if training:
            # Select only the kept patches and run transformer:
            after_patch_kept = torch.gather(
                after_patch, 1,
                idxs.unsqueeze(-1).expand(-1, -1, after_patch.size(-1))
            )
            after_enc_kept = self.encoder.run_transformer(after_patch_kept)  # [B, D, top_k, embed_dim]
            after_ctx_kept = after_enc_kept.transpose(1, 2).reshape(B, top_k, self.depth, self.embed_dim)

            # Prepare a full tensor filled with the learned drop token.
            drop_token_flat = self.drop_token[0, 0]  # shape: [D*embed_dim]
            # Reshape drop token to [D, embed_dim]
            drop_token_reshaped = drop_token_flat.reshape(self.depth, self.embed_dim)
            full_after = drop_token_reshaped.expand(B, P, self.depth, self.embed_dim).clone()

            # Scatter the kept patch features back into their original positions.
            full_after.scatter_(1,
                idxs.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.depth, self.embed_dim),
                after_ctx_kept
            )
            after_ctx = full_after  # [B, P, D, embed_dim]
        else:
            after_enc = self.encoder.run_transformer(after_patch)  # [B, D, P, embed_dim]
            after_ctx = after_enc.transpose(1, 2).reshape(B, P, self.depth, self.embed_dim)

        # --- Interleave along the depth dimension ---
        # Both before_ctx and after_ctx are [B, P, D, embed_dim]
        # Stack along a new dim (dim=3) to get [B, P, D, 2, embed_dim]
        combined = torch.stack([before_ctx, after_ctx], dim=3)
        # Reshape so that the depth channels are interleaved: [B, P, D*2, embed_dim]
        combined = combined.reshape(B, P, self.depth * 2, self.embed_dim)
        # Finally, flatten the last two dimensions to obtain [B, P, 2*D*embed_dim]
        src = combined.reshape(B, P, self.depth * 2 * self.embed_dim)

        return src


    def forward(self, x_before: torch.Tensor, x_after: torch.Tensor):
        """
        Top-level forward for either training or eval.

        Returns:
          In eval mode (training=False):
            {
              "la": Tensor [B, num_eval_tokens, action_dim],
              "la_flat": Tensor [B, num_eval_tokens * action_dim]
            }

          In train mode:
            {
              "la": Tensor [B, num_tokens, action_dim],
            }
        """
        device = x_before.device
        B = x_before.shape[0]

        if not self.training:

            num_tokens = self.num_eval_tokens

            src = self._encode_vision_context(x_before, x_after, training=False)
            # src shape: [B, P, d_context] where d_context = depth*embed_dim*2

            memory = src

            tgt_so_far = self.mask_token.expand(B, 1, -1).clone()

            out_tokens = []
            for step_i in range(num_tokens):
                T = tgt_so_far.shape[1]
                causal_mask = self._make_causal_mask(T, device=device)

                decoder_out = self.transformer.decoder(
                    tgt=tgt_so_far,
                    memory=memory,
                    tgt_mask=causal_mask,
                )  # [B, T, d_context]

                last_token = decoder_out[:, -1:, :]  # [B, 1, d_context]
                action = self.action_head(last_token)  # [B, 1, action_dim]
                out_tokens.append(action)

                new_embed = last_token
                tgt_so_far = torch.cat([tgt_so_far, new_embed], dim=1)

            la = torch.cat(out_tokens, dim=1)  # [B, num_tokens, action_dim]
            la_flat = la.view(B, -1)

            return TensorDict(
                {"la": la, "la_flat": la_flat},
                batch_size=B,
            )

        else:

            num_tokens = torch.randint(
                low=self.min_tokens,
                high=self.max_tokens + 1,
                size=(1,),
                device=device
            ).item()

            out_tokens = []
            tgt_so_far = self.mask_token.expand(B, 1, -1).clone()

            for step_i in range(num_tokens):
                # Re-encode vision context with augmentation and patch keep/drop logic:
                src = self._encode_vision_context(x_before, x_after, training=True)
                memory = src

                T = tgt_so_far.shape[1]
                causal_mask = self._make_causal_mask(T, device=src.device)

                decoder_out = self.transformer.decoder(
                    tgt=tgt_so_far,
                    memory=memory,
                    tgt_mask=causal_mask,
                )  # [B, T, d_context]

                last_token = decoder_out[:, -1:, :]
                action = self.action_head(last_token)
                out_tokens.append(action)

                new_embed = last_token
                tgt_so_far = torch.cat([tgt_so_far, new_embed], dim=1)

            la = torch.cat(out_tokens, dim=1)  # [B, num_tokens, action_dim]
            return TensorDict({"la": la}, batch_size=B)

    def label(self, batch: TensorDict):
        """
        Mutates the batch with predicted action embeddings,
        storing them under batch["la"] (or ["la_flat"] in eval).
        """
        x_before = batch["obs"][:, -2]
        x_after  = batch["obs"][:, -1]
        action_dict = self(x_before, x_after)
        batch.update(action_dict)
        del action_dict
        torch.cuda.empty_cache()

    @torch.no_grad()
    def label_chunked(
        self,
        data: TensorDict,
        chunksize: int = 128,
    ) -> TensorDict:
        """
        Example method for labeling large datasets in chunks
        (in inference mode).
        """
        self.encoder.eval()
        device = next(self.encoder.parameters()).device

        out_dicts = []

        def _label(data_batch: TensorDict):
            after_obs = normalize_obs(data_batch["obs"][:, -2].to(device))
            before_obs = normalize_obs(data_batch["obs"][:, -1].to(device))
            preds = self(before_obs, after_obs).to(data_batch.device)
            return preds

        for data_batch in data.split(chunksize):
            out = _label(data_batch)
            out_dicts.append(out.cpu())
            del out
            torch.cuda.empty_cache()

        action_dicts = torch.cat(out_dicts)
        data.update(action_dicts)
        del action_dicts
        torch.cuda.empty_cache()
        return data


def create_dynamics_models(
    model_cfg: ModelConfig, state_dicts: dict | None = None,
    embed_dim: int = 128,
) -> tuple[IDM, WorldModel]:
    obs_depth = 3


    wm = WorldModel(
        embed_dim=embed_dim,
        action_dim=model_cfg.la_dim,
    ).to(DEVICE)

    encoder = wm.get_encoder()

    ema_encoder = wm.get_ema_encoder()


    idm = IDM(
        encoder=encoder,
        ema_encoder=ema_encoder,
        action_dim=model_cfg.la_dim,
    ).to(DEVICE)

    if state_dicts is not None:
        idm.load_state_dict(state_dicts["idm"])
        wm.load_state_dict(state_dicts["wm"])

    return idm, wm

total_steps = cfg.stage1.steps

INITIAL_LR = 1e-7
PEAK_LR = 1e-4
FINAL_LR = 3e-5

def lr_lambda(step):
    warmup_steps = int(0.15 * total_steps)
    if step < warmup_steps:
        # Linear warmup from INITIAL_LR to PEAK_LR
        return (PEAK_LR - INITIAL_LR) / INITIAL_LR * (step / warmup_steps) + 1
    else:
        # Cosine decay from PEAK_LR to FINAL_LR
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress + math.pi))
        return (PEAK_LR + (FINAL_LR - PEAK_LR) * cosine_decay) / INITIAL_LR


def get_unique_params(model1, model2):
    # I wrote this very late at night, this is very stupid.
    params1 = set(model1.parameters())
    params2 = set(model2.parameters())
    return list(params1.union(params2))

from doy import loop

idm, wm = create_dynamics_models(cfg.model, embed_dim=128)

optimizer = torch.optim.AdamW(get_unique_params(idm, wm), lr=INITIAL_LR, weight_decay=0.0)
scheduler = LambdaLR(optimizer, lr_lambda)
start_momentum = 0.99


global_progress = 0

def train_step(index):
    global total_steps, start_momentum, global_progress
    momentum = start_momentum

    idm.train()
    wm.train()
    batch = next(train_iter)

    idm.label(batch)
    wm_loss_tensor = wm.label(batch)

    wm_loss = wm_loss_tensor.mean()

    # Combine losses
    loss = wm_loss

    optimizer.zero_grad()  # Reset gradients after updating

    loss.backward()

    # grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(get_unique_params(idm, wm), 0.2)

    optimizer.step()
    wm.update_ema(momentum)

    if index % 250 == 0:
        print(f"Step {index}, wm loss: {wm_loss.item():.6f}")

    logger(
        index,
        # grad_norm=grad_norm,
        wm_loss=wm_loss.item(),
        loss=loss.item(),
        global_step=index * cfg.stage1.bs,
        ema_momentum=momentum,
        lr=optimizer.param_groups[0]['lr'],
        weight_decay=optimizer.param_groups[0]['weight_decay'],
    )

    del batch
    del loss
    del wm_loss
    wm.zero_grad()
    idm.zero_grad()
    torch.cuda.empty_cache()
    gc.collect()


for step in range(cfg.stage1.steps + 1):
    train_step(step)

    scheduler.step()
