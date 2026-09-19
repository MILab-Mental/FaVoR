import re
import torch
import torch.nn as nn
from collections import Counter, defaultdict


def load_checkpoint_state_dict(path):
    """
    Load emotion2vec checkpoint.

    Try weights_only=True first.
    If it fails, fallback to weights_only=False for trusted local checkpoints.
    """
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict):
        print("=" * 80)
        print("Top-level checkpoint keys:")
        for k in list(ckpt.keys())[:30]:
            print(" ", k)
        print("=" * 80)

        for key in ["model", "state_dict", "module", "net"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]

        return ckpt

    raise ValueError(f"Unknown checkpoint format: {path}")


def inspect_checkpoint(state, max_samples=80):
    """
    Robustly inspect emotion2vec / data2vec_multi checkpoint.
    """
    print("=" * 80)
    print("Inspecting checkpoint")
    print("=" * 80)

    if not isinstance(state, dict):
        raise ValueError("State is not a dict.")

    # ------------------------------------------------------------------
    # 1. Top-level prefixes
    # ------------------------------------------------------------------
    top_prefixes = Counter(k.split(".")[0] for k in state.keys())
    second_prefixes = Counter(".".join(k.split(".")[:2]) for k in state.keys())

    print("\n[Top-level prefixes]")
    for prefix, count in top_prefixes.most_common(20):
        print(f"  {prefix}: {count}")

    print("\n[Second-level prefixes]")
    for prefix, count in second_prefixes.most_common(50):
        print(f"  {prefix}: {count}")

    # ------------------------------------------------------------------
    # 2. CNN analysis
    # ------------------------------------------------------------------
    print("\n[CNN analysis]")

    conv_pattern = re.compile(
        r"conv_layers\.(\d+)\.(\d+)(?:\.\d+)?\.(weight|bias)"
    )

    conv_layer_ids = set()
    norm_layer_ids = set()
    conv_weight_shapes = {}

    for k, v in state.items():
        if not torch.is_tensor(v):
            continue

        m = conv_pattern.search(k)
        if not m:
            continue

        layer_idx = int(m.group(1))
        sub_idx = int(m.group(2))
        param_type = m.group(3)

        conv_layer_ids.add(layer_idx)

        if sub_idx == 2:
            norm_layer_ids.add(layer_idx)

        if sub_idx == 0 and param_type == "weight" and v.dim() == 3:
            conv_weight_shapes[layer_idx] = tuple(v.shape)

    print(f"  CNN layers found: {sorted(conv_layer_ids)}")
    print(f"  CNN layers with norm: {sorted(norm_layer_ids)}")

    for layer_idx in sorted(conv_weight_shapes.keys()):
        out_ch, in_ch, kernel_size = conv_weight_shapes[layer_idx]
        print(
            f"    layer {layer_idx}: out_ch={out_ch}, in_ch={in_ch}, kernel_size={kernel_size}"
        )

    if len(conv_layer_ids) > 0:
        if len(norm_layer_ids) == len(conv_layer_ids):
            print("  CNN norm mode likely: layer_norm (norm in every layer)")
        elif len(norm_layer_ids) == 1 and 0 in norm_layer_ids:
            print("  CNN norm mode likely: default (norm only in first layer)")
        else:
            print("  CNN norm mode: UNKNOWN, please inspect manually")

    # ------------------------------------------------------------------
    # 3. Transformer encoder candidates
    # ------------------------------------------------------------------
    print("\n[Transformer encoder candidates]")

    skip_keywords = [
        "conv_layers",
        "decoder",
        "prenet",
        "image",
        "text",
        "alibi",
        "extra_tokens",
        "quantizer",
        "mask",
    ]

    key_patterns = [
        r"(q_proj|k_proj|v_proj|out_proj)\.weight$",
        r"(in_proj_weight|in_proj_bias)$",
        r"(fc1|fc2)\.weight$",
        r"(self_attn_layer_norm|final_layer_norm|norm1|norm2|layer_norm)\.weight$",
    ]

    layer_pattern = re.compile(r"(?P<prefix>.*\.)?(?:layers|blocks)\.(?P<idx>\d+)\.")

    candidate_keys = []
    layer_prefixes = defaultdict(set)

    for k, v in state.items():
        if not torch.is_tensor(v):
            continue

        if any(s in k.lower() for s in skip_keywords):
            continue

        matched = False
        for pat in key_patterns:
            if re.search(pat, k):
                matched = True
                break

        if not matched:
            continue

        candidate_keys.append((k, tuple(v.shape)))

        m = layer_pattern.search(k)
        if m:
            prefix = m.group("prefix") or "<root>."
            idx = int(m.group("idx"))
            layer_prefixes[prefix].add(idx)

    if len(layer_prefixes) == 0:
        print("  No transformer encoder candidates found.")
    else:
        for prefix, idxs in layer_prefixes.items():
            max_idx = max(idxs)
            depth = max_idx + 1
            print(f"  prefix: {prefix}")
            print(f"    layer ids: {sorted(idxs)}")
            print(f"    inferred depth: {depth}")

    print("\n[Sample transformer candidate keys]")
    if len(candidate_keys) == 0:
        print("  No candidate keys found.")
    else:
        for k, shape in candidate_keys[:max_samples]:
            print(f"  {k}: {shape}")

    # ------------------------------------------------------------------
    # 4. Extra / positional / alibi / utterance keys
    # ------------------------------------------------------------------
    print("\n[Extra / positional / alibi / cls / utterance keys]")

    extra_keywords = [
        "extra_tokens",
        "cls_token",
        "utterance",
        "alibi",
        "pos",
        "mask",
        "decoder",
        "prenet",
    ]

    extra_keys = []
    for k, v in state.items():
        if not torch.is_tensor(v):
            continue
        if any(s in k.lower() for s in extra_keywords):
            extra_keys.append((k, tuple(v.shape)))

    for k, shape in extra_keys[:max_samples]:
        print(f"  {k}: {shape}")

    if len(extra_keys) == 0:
        print("  No extra keys found.")

    # ------------------------------------------------------------------
    # 5. If no transformer candidates, dump non-CNN keys
    # ------------------------------------------------------------------
    if len(candidate_keys) == 0:
        print("\n[WARNING] No transformer encoder candidates found.")
        print("Printing first 200 non-CNN keys to help locate the real encoder.")

        count = 0
        for k, v in state.items():
            if not torch.is_tensor(v):
                continue
            if "conv_layers" in k:
                continue

            print(f"  {k}: {tuple(v.shape)}")

            count += 1
            if count >= 200:
                break

    print("=" * 80)


def _strip_prefixes(key, prefixes):
    for p in prefixes:
        if key.startswith(p):
            return key[len(p):]
    return key


def load_extractor_weights(model_extractor, state, min_load_ratio=0.8):
    target_state = model_extractor.state_dict()

    candidate_prefixes = [
        "d2v_model.modality_encoders.AUDIO.local_encoder.",
        "modality_encoders.AUDIO.local_encoder.",
        "d2v_model.feature_extractor.",
        "model.feature_extractor.",
        "module.feature_extractor.",
        "feature_extractor.",
        "encoder.feature_extractor.",
        "backbone.feature_extractor.",
    ]

    matched = {}
    skipped = []

    # Supports:
    #   conv_layers.0.0.weight
    #   conv_layers.0.2.weight
    #   conv_layers.0.2.1.weight
    pattern = re.compile(r"conv_layers\.(\d+)\.(\d+)(?:\.\d+)?\.(weight|bias)")

    for k, v in state.items():
        if not torch.is_tensor(v):
            continue

        stripped = _strip_prefixes(k, candidate_prefixes)
        m = pattern.search(stripped)

        if not m:
            continue

        layer_idx = int(m.group(1))
        sub_idx = int(m.group(2))
        param_type = m.group(3)

        if sub_idx == 0:
            new_key = f"conv_blocks.{layer_idx}.conv.{param_type}"
        elif sub_idx == 2:
            new_key = f"conv_blocks.{layer_idx}.layer_norm.{param_type}"
        else:
            skipped.append((k, tuple(v.shape), "unknown_sub_module"))
            continue

        if new_key in target_state and target_state[new_key].shape == v.shape:
            matched[new_key] = v
        else:
            skipped.append((
                k,
                tuple(v.shape),
                target_state.get(new_key).shape if new_key in target_state else "KEY_NOT_FOUND"
            ))

    missing = [k for k in target_state.keys() if k not in matched]

    target_state.update(matched)
    model_extractor.load_state_dict(target_state, strict=False)

    loaded_numel = sum(v.numel() for v in matched.values())
    total_numel = sum(p.numel() for p in model_extractor.parameters())
    ratio = loaded_numel / max(1, total_numel)

    print("[Extractor transfer]")
    print(f"  matched tensors: {len(matched)}")
    print(f"  skipped tensors: {len(skipped)}")
    print(f"  missing target tensors: {len(missing)}")
    print(f"  loaded params: {loaded_numel / 1e6:.2f}M / {total_numel / 1e6:.2f}M")
    print(f"  load ratio: {ratio * 100:.2f}%")

    if missing:
        print("  first missing target keys:")
        for item in missing[:10]:
            print("   ", item)

    if skipped:
        print("  first skipped:")
        for item in skipped[:10]:
            print("   ", item)

    assert ratio >= min_load_ratio, (
        f"Extractor load ratio {ratio * 100:.2f}% is below threshold "
        f"{min_load_ratio * 100:.2f}%. Check checkpoint keys and extractor structure."
    )

    return model_extractor


def load_feature_projection_weights(model, state):
    """
    Load emotion2vec_plus_large feature projection weights.

    Source:
      d2v_model.modality_encoders.AUDIO.project_features.1.weight
      d2v_model.modality_encoders.AUDIO.project_features.1.bias
      d2v_model.modality_encoders.AUDIO.project_features.2.weight
      d2v_model.modality_encoders.AUDIO.project_features.2.bias

    Target:
      feature_norm.weight
      feature_norm.bias
      post_extraction_mapper.weight
      post_extraction_mapper.bias
    """
    prefix = "d2v_model.modality_encoders.AUDIO.project_features."

    norm_w_key = prefix + "1.weight"
    norm_b_key = prefix + "1.bias"
    proj_w_key = prefix + "2.weight"
    proj_b_key = prefix + "2.bias"

    required_keys = [norm_w_key, norm_b_key, proj_w_key, proj_b_key]

    for k in required_keys:
        if k not in state:
            raise KeyError(f"Missing feature projection key in checkpoint: {k}")

    feature_norm_state = {
        "weight": state[norm_w_key],
        "bias": state[norm_b_key],
    }

    mapper_state = {
        "weight": state[proj_w_key],
        "bias": state[proj_b_key],
    }

    assert model.feature_norm.weight.shape == feature_norm_state["weight"].shape
    assert model.feature_norm.bias.shape == feature_norm_state["bias"].shape
    assert model.post_extraction_mapper.weight.shape == mapper_state["weight"].shape
    assert model.post_extraction_mapper.bias.shape == mapper_state["bias"].shape

    model.feature_norm.load_state_dict(feature_norm_state)
    model.post_extraction_mapper.load_state_dict(mapper_state)

    print("[Feature projection transfer]")
    print("  feature_norm: loaded")
    print("  post_extraction_mapper: loaded")

    return model


def load_encoder_weights(model_encoder, state, min_load_ratio=0.9):
    """
    Load emotion2vec_plus_large main Transformer encoder weights.

    Source keys:
      d2v_model.blocks.{i}.norm1.*
      d2v_model.blocks.{i}.attn.qkv.*
      d2v_model.blocks.{i}.attn.proj.*
      d2v_model.blocks.{i}.norm2.*
      d2v_model.blocks.{i}.mlp.fc1.*
      d2v_model.blocks.{i}.mlp.fc2.*

    Target keys:
      layers.{i}.norm1.*
      layers.{i}.attn.qkv.*
      layers.{i}.attn.proj.*
      layers.{i}.norm2.*
      layers.{i}.mlp.fc1.*
      layers.{i}.mlp.fc2.*
    """
    target_state = model_encoder.state_dict()

    matched = {}
    skipped = []

    source_prefix = "d2v_model.blocks."
    target_prefix = "layers."

    for k, v in state.items():
        if not torch.is_tensor(v):
            continue

        if not k.startswith(source_prefix):
            continue

        new_key = target_prefix + k[len(source_prefix):]

        if new_key in target_state and target_state[new_key].shape == v.shape:
            matched[new_key] = v
        else:
            skipped.append((
                k,
                tuple(v.shape),
                target_state.get(new_key).shape if new_key in target_state else "KEY_NOT_FOUND"
            ))

    missing = [k for k in target_state.keys() if k not in matched]

    target_state.update(matched)
    model_encoder.load_state_dict(target_state, strict=False)

    loaded_numel = sum(v.numel() for v in matched.values())
    total_numel = sum(p.numel() for p in model_encoder.parameters())
    ratio = loaded_numel / max(1, total_numel)

    print("[Encoder transfer]")
    print(f"  matched tensors: {len(matched)}")
    print(f"  skipped tensors: {len(skipped)}")
    print(f"  missing target tensors: {len(missing)}")
    print(f"  loaded params: {loaded_numel / 1e6:.2f}M / {total_numel / 1e6:.2f}M")
    print(f"  load ratio: {ratio * 100:.2f}%")

    if missing:
        print("  first missing target keys:")
        for item in missing[:10]:
            print("   ", item)

    if skipped:
        print("  first skipped:")
        for item in skipped[:10]:
            print("   ", item)

    assert ratio >= min_load_ratio, (
        f"Encoder load ratio {ratio * 100:.2f}% is below threshold "
        f"{min_load_ratio * 100:.2f}%. Check checkpoint keys and encoder structure."
    )

    return model_encoder

# def load_encoder_weights(model_encoder, state, min_load_ratio=0.8):
#     """
#     Load emotion2vec Transformer encoder weights.

#     Our target encoder uses fairseq-like names:
#       layers.0.self_attn.q_proj.weight
#       layers.0.self_attn.k_proj.weight
#       layers.0.self_attn.v_proj.weight
#       layers.0.self_attn.out_proj.weight
#       layers.0.fc1.weight
#       layers.0.fc2.weight
#       layers.0.self_attn_layer_norm.weight
#       layers.0.final_layer_norm.weight

#     emotion2vec / data2vec checkpoints often use:
#       encoder.layers.0.self_attn.q_proj.weight
#       ...
#     """
#     target_state = model_encoder.state_dict()

#     # candidate_prefixes = [
#     #     "module.encoder.",
#     #     "encoder.",
#     #     "backbone.encoder.",
#     #     "backbone.",
#     #     "modality_encoders.AUDIO.local_encoder.",
#     # ]
#     candidate_prefixes = [
#         "d2v_model.encoder.",
#         "model.encoder.",
#         "module.encoder.",
#         "encoder.",
#         "backbone.encoder.",
#         "backbone.",
#         "d2v_model.modality_encoders.AUDIO.local_encoder.",
#         "modality_encoders.AUDIO.local_encoder.",
#     ]

#     matched = {}
#     skipped = []

#     for k, v in state.items():
#         stripped = _strip_prefixes(k, candidate_prefixes)

#         # We only want encoder layer weights.
#         if not stripped.startswith("layers."):
#             continue

#         if stripped in target_state and target_state[stripped].shape == v.shape:
#             matched[stripped] = v
#         else:
#             skipped.append((
#                 k,
#                 tuple(v.shape),
#                 target_state.get(stripped).shape if stripped in target_state else "KEY_NOT_FOUND"
#             ))

#     target_state.update(matched)
#     model_encoder.load_state_dict(target_state)

#     loaded_numel = sum(v.numel() for v in matched.values())
#     total_numel = sum(p.numel() for p in model_encoder.parameters())
#     ratio = loaded_numel / max(1, total_numel)

#     print("[Encoder transfer]")
#     print(f"  matched tensors: {len(matched)}")
#     print(f"  skipped tensors: {len(skipped)}")
#     print(f"  loaded params: {loaded_numel / 1e6:.2f}M / {total_numel / 1e6:.2f}M")
#     print(f"  load ratio: {ratio * 100:.2f}%")

#     if skipped:
#         print("  first skipped:")
#         for item in skipped[:10]:
#             print("   ", item)

#     assert ratio >= min_load_ratio, (
#         f"Encoder load ratio {ratio * 100:.2f}% is below threshold "
#         f"{min_load_ratio * 100:.2f}%. Check checkpoint keys and encoder structure."
#     )

#     return model_encoder


# ---------------------------------------------------------------------------
# FAVOR checkpoint adapters
# ---------------------------------------------------------------------------

import logging

LOGGER = logging.getLogger(__name__)


def _official_emotion2vec_mappings(backbone, state):
    """Return per-component target/source mappings for a raw official checkpoint."""
    audio = "d2v_model.modality_encoders.AUDIO."
    components = {}

    extractor = {}
    pattern = re.compile(r"conv_layers\.(\d+)\.(\d+)(?:\.\d+)?\.(weight|bias)$")
    for source_key, value in state.items():
        if not source_key.startswith(audio + "local_encoder.") or not torch.is_tensor(value):
            continue
        match = pattern.search(source_key)
        if not match:
            continue
        layer, submodule, parameter = match.groups()
        if submodule == "0":
            target_key = f"feature_extractor.conv_blocks.{layer}.conv.{parameter}"
        elif submodule == "2":
            target_key = f"feature_extractor.conv_blocks.{layer}.layer_norm.{parameter}"
        else:
            continue
        extractor[target_key] = (source_key, value)
    components["local_encoder"] = extractor

    projection_keys = {
        "feature_norm.weight": audio + "project_features.1.weight",
        "feature_norm.bias": audio + "project_features.1.bias",
        "post_extraction_mapper.weight": audio + "project_features.2.weight",
        "post_extraction_mapper.bias": audio + "project_features.2.bias",
    }
    components["feature_projection"] = {
        target: (source, state[source]) for target, source in projection_keys.items() if source in state
    }

    relative = {}
    relative_pattern = re.compile(
        re.escape(audio) + r"relative_positional_encoder\.(\d+)\.0\.(weight|bias)$"
    )
    for source_key, value in state.items():
        match = relative_pattern.match(source_key)
        if match:
            layer, parameter = match.groups()
            target_key = f"relative_positional_encoder.layers.{int(layer) - 1}.conv.{parameter}"
            relative[target_key] = (source_key, value)
    components["relative_position"] = relative

    modality = {}
    modality_prefix = audio + "context_encoder."
    for source_key, value in state.items():
        if not source_key.startswith(modality_prefix) or not torch.is_tensor(value):
            continue
        suffix = source_key[len(modality_prefix):]
        if suffix.startswith("blocks."):
            suffix = "layers." + suffix[len("blocks."):]
        modality[f"modality_context_encoder.{suffix}"] = (source_key, value)
    components["modality_context"] = modality

    components["extra_tokens"] = {
        "extra_tokens": (audio + "extra_tokens", state[audio + "extra_tokens"])
    } if audio + "extra_tokens" in state else {}
    components["alibi_scale"] = {
        "alibi_scale": (audio + "alibi_scale", state[audio + "alibi_scale"])
    } if audio + "alibi_scale" in state else {}

    global_encoder = {}
    for source_key, value in state.items():
        if source_key.startswith("d2v_model.blocks.") and torch.is_tensor(value):
            suffix = source_key[len("d2v_model.blocks."):]
            global_encoder[f"context_encoder.layers.{suffix}"] = (source_key, value)
        elif source_key.startswith("d2v_model.norm.") and torch.is_tensor(value):
            suffix = source_key[len("d2v_model.norm."):]
            global_encoder[f"context_encoder.norm.{suffix}"] = (source_key, value)
    components["global_encoder"] = global_encoder
    return components


def load_official_emotion2vec_backbone(backbone, state, min_parameter_ratio=0.95):
    """Load and audit every parameter on the official emotion2vec inference path."""
    if getattr(backbone, "mode", None) != "emotion2vec":
        raise ValueError(
            "A full official emotion2vec checkpoint requires model.backbone.mode=emotion2vec"
        )
    target_state = backbone.state_dict()
    components = _official_emotion2vec_mappings(backbone, state)
    compatible = {}
    consumed_sources = set()
    reports = []

    component_prefixes = {
        "local_encoder": ("feature_extractor.",),
        "feature_projection": ("feature_norm.", "post_extraction_mapper."),
        "relative_position": ("relative_positional_encoder.",),
        "modality_context": ("modality_context_encoder.",),
        "extra_tokens": ("extra_tokens",),
        "alibi_scale": ("alibi_scale",),
        "global_encoder": ("context_encoder.",),
    }
    for name, mapping in components.items():
        target_keys = [
            key for key in target_state
            if any(key == prefix or key.startswith(prefix) for prefix in component_prefixes[name])
        ]
        loaded_target = 0
        loaded_source = 0
        mismatches = []
        for target_key, (source_key, value) in mapping.items():
            if target_key in target_state and target_state[target_key].shape == value.shape:
                compatible[target_key] = value
                consumed_sources.add(source_key)
                loaded_target += value.numel()
                loaded_source += value.numel()
            else:
                mismatches.append((source_key, tuple(value.shape), target_key,
                                   tuple(target_state[target_key].shape) if target_key in target_state else None))
        target_total = sum(target_state[key].numel() for key in target_keys)
        source_total = sum(value.numel() for _, value in mapping.values())
        reports.append((name, loaded_target, target_total, loaded_source, source_total, mismatches))

    official_source = {
        key: value for key, value in state.items()
        if torch.is_tensor(value) and (
            key.startswith("d2v_model.modality_encoders.AUDIO.")
            or key.startswith("d2v_model.blocks.")
            or key.startswith("d2v_model.norm.")
        )
    }
    loaded_target_total = sum(value.numel() for value in compatible.values())
    target_total = sum(value.numel() for value in target_state.values())
    source_total = sum(value.numel() for value in official_source.values())
    loaded_source_total = sum(state[key].numel() for key in consumed_sources)
    target_ratio = loaded_target_total / max(1, target_total)
    source_ratio = loaded_source_total / max(1, source_total)

    LOGGER.info("emotion2vec checkpoint coverage by component:")
    for name, loaded_target, target_count, loaded_source, source_count, mismatches in reports:
        LOGGER.info(
            "  %-20s target=%7.2f%% (%8.3fM/%8.3fM) source=%7.2f%% (%8.3fM/%8.3fM)%s",
            name,
            100.0 * loaded_target / max(1, target_count), loaded_target / 1e6, target_count / 1e6,
            100.0 * loaded_source / max(1, source_count), loaded_source / 1e6, source_count / 1e6,
            f" mismatched={len(mismatches)}" if mismatches else "",
        )
    skipped_sources = [key for key in official_source if key not in consumed_sources]
    missing_targets = [key for key in target_state if key not in compatible]
    LOGGER.info(
        "emotion2vec target coverage: %.2f%% (%.3fM/%.3fM parameters); missing tensors=%d",
        target_ratio * 100, loaded_target_total / 1e6, target_total / 1e6, len(missing_targets),
    )
    LOGGER.info(
        "emotion2vec source coverage: %.2f%% (%.3fM/%.3fM encoder parameters); skipped tensors=%d",
        source_ratio * 100, loaded_source_total / 1e6, source_total / 1e6, len(skipped_sources),
    )
    excluded = {
        key: value for key, value in state.items()
        if torch.is_tensor(value) and key not in official_source
    }
    if excluded:
        LOGGER.info(
            "Excluded non-encoder checkpoint tensors: %d (%.3fM parameters), first keys=%s",
            len(excluded), sum(value.numel() for value in excluded.values()) / 1e6,
            list(excluded)[:10],
        )
    if missing_targets:
        LOGGER.warning("First missing emotion2vec target tensors: %s", missing_targets[:10])
    if skipped_sources:
        LOGGER.warning("First unused official encoder tensors: %s", skipped_sources[:10])
    if target_ratio < min_parameter_ratio or source_ratio < min_parameter_ratio:
        raise RuntimeError(
            f"Incomplete emotion2vec transfer: target={target_ratio:.2%}, source={source_ratio:.2%}, "
            f"required={min_parameter_ratio:.2%}"
        )
    backbone.load_state_dict(compatible, strict=False)
    return {
        "target_ratio": target_ratio,
        "source_ratio": source_ratio,
        "missing_targets": missing_targets,
        "skipped_sources": skipped_sources,
    }


def init_from_emotion2vec(model, path, cfg):
    """Initialize a FAVOR AudioJEPA model from an emotion2vec checkpoint."""
    state = load_checkpoint_state_dict(path)
    init_cfg = cfg.get("init", {})
    if getattr(model.encoder, "mode", None) == "emotion2vec":
        load_official_emotion2vec_backbone(
            model.encoder,
            state,
            min_parameter_ratio=float(init_cfg.get("min_load_ratio", 0.95)),
        )
        model.sync_target_encoder()
        return
    if init_cfg.get("load_extractor", True):
        load_extractor_weights(
            model.encoder.feature_extractor,
            state,
            min_load_ratio=float(init_cfg.get("min_extractor_load_ratio", 0.95)),
        )
    load_feature_projection_weights(model.encoder, state)
    if init_cfg.get("load_encoder", True):
        load_encoder_weights(
            model.encoder.context_encoder,
            state,
            min_load_ratio=float(init_cfg.get("min_encoder_load_ratio", 0.95)),
        )
    model.sync_target_encoder()


def save_audio_pretrain_checkpoint(path, *, model, optimizer, step, epoch, args, history=None):
    raw = model.module if hasattr(model, "module") else model
    torch.save({
        "schema_version": 2,
        "modality": "audio",
        "stage": "pretrain",
        "step": int(step),
        "epoch": int(epoch),
        "encoder": raw.encoder.state_dict(),
        "target_encoder": raw.target_encoder.state_dict(),
        "predictor": raw.predictor.state_dict(),
        "optimizer": optimizer.state_dict(),
        "history": list(history or []),
        "args": args,
    }, path)


def restore_audio_pretrain_checkpoint(path, model, optimizer=None, strict=True):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not all(key in checkpoint for key in ("encoder", "target_encoder", "predictor")):
        raise ValueError(f"Not a FAVOR audio pre-training checkpoint: {path}")
    model.encoder.load_state_dict(checkpoint["encoder"], strict=strict)
    model.target_encoder.load_state_dict(checkpoint["target_encoder"], strict=strict)
    model.predictor.load_state_dict(checkpoint["predictor"], strict=strict)
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def _strip_prefix(key, prefixes):
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
                break
    return key


def audio_backbone_state(checkpoint):
    """Extract a backbone from FAVOR, legacy AEmo-JEPA, or raw state dicts."""
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("encoder"), dict):
        return checkpoint["encoder"]
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise ValueError("Unsupported audio checkpoint format")
    allowed = (
        "feature_extractor.", "feature_norm.", "post_extraction_mapper.",
        "pos_embed_encoder.", "relative_positional_encoder.",
        "modality_context_encoder.", "extra_tokens", "alibi_scale",
        "context_encoder.",
    )
    extracted = {}
    for key, value in state.items():
        clean = _strip_prefix(key, ("module.", "backbone.", "model.", "encoder."))
        if clean.startswith(allowed):
            extracted[clean] = value
    return extracted or state


def load_audio_backbone(backbone, checkpoint_path, min_parameter_ratio=0.95):
    if not checkpoint_path:
        LOGGER.warning("No audio backbone checkpoint supplied; using random initialization")
        return None
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw_state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if isinstance(raw_state, dict) and any(
        key.startswith("d2v_model.blocks.") for key in raw_state
    ):
        # Raw emotion2vec/data2vec checkpoints use fairseq-style names rather
        # than FAVOR's AudioBackbone names. Reuse the same explicit mapping as
        # pretrain_a initialization so finetune_a can start directly from an
        # emotion2vec checkpoint without first producing an AEmo-JEPA file.
        if getattr(backbone, "mode", None) == "emotion2vec":
            load_official_emotion2vec_backbone(backbone, raw_state, min_parameter_ratio)
        else:
            LOGGER.warning(
                "Loading only CNN/projection/global blocks because backbone.mode=%s. "
                "Use backbone.mode=emotion2vec for the complete official encoder.",
                getattr(backbone, "mode", None),
            )
            load_extractor_weights(
                backbone.feature_extractor,
                raw_state,
                min_load_ratio=min_parameter_ratio,
            )
            load_feature_projection_weights(backbone, raw_state)
            load_encoder_weights(
                backbone.context_encoder,
                raw_state,
                min_load_ratio=min_parameter_ratio,
            )
        LOGGER.info("Initialized audio backbone directly from emotion2vec checkpoint %s", checkpoint_path)
        return checkpoint
    source = audio_backbone_state(checkpoint)
    current = backbone.state_dict()
    compatible = {
        key: value for key, value in source.items()
        if key in current and current[key].shape == value.shape
    }
    loaded = sum(value.numel() for value in compatible.values())
    total = sum(value.numel() for value in current.values())
    ratio = loaded / max(1, total)
    if ratio < min_parameter_ratio:
        missing = [key for key in current if key not in compatible]
        raise RuntimeError(
            f"Audio backbone coverage {ratio:.2%} is below {min_parameter_ratio:.2%}; "
            f"first missing keys: {missing[:10]}"
        )
    result = backbone.load_state_dict(compatible, strict=False)
    LOGGER.info(
        "Loaded audio backbone %.2f%% (%d/%d tensors) from %s; missing=%d",
        ratio * 100, len(compatible), len(current), checkpoint_path, len(result.missing_keys),
    )
    return checkpoint
