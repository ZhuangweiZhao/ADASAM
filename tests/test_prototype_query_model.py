import torch

from adasam.losses import PrototypeQuerySemanticLoss
from adasam.model import PrototypeConditionedSemanticQueryDecoder, PrototypeQueryConfig


def test_prototype_conditioned_query_shapes_and_gradients():
    config = PrototypeQueryConfig(
        embed_dim=32, num_queries=4, num_layers=2, num_heads=4, ffn_dim=64
    )
    model = PrototypeConditionedSemanticQueryDecoder(config)
    query = torch.randn(1, 32, 8, 8)
    support = torch.randn(2, 32, 8, 8)
    support_masks = torch.zeros(2, 8, 8)
    support_masks[:, 2:6, 2:6] = 1
    output = model(query, support, support_masks)
    assert output.query_logits.shape == (1, 4)
    assert output.query_mask_logits.shape == (1, 4, 8, 8)
    assert output.semantic_logits.shape == (1, 8, 8)
    assert output.conditioned_queries.shape == (1, 4, 32)
    assert output.prototype.shape == (1, 32)
    assert len(output.auxiliary) == 1

    target = support_masks[:1]
    losses = PrototypeQuerySemanticLoss()(output, target)
    losses["loss"].backward()
    assert model.query_embed.weight.grad is not None
    assert model.prototype_film.weight.grad is not None


def test_semantic_probability_is_bounded():
    model = PrototypeConditionedSemanticQueryDecoder(
        PrototypeQueryConfig(embed_dim=16, num_queries=2, num_layers=1, num_heads=4, ffn_dim=32)
    )
    output = model(
        torch.randn(1, 16, 4, 4),
        torch.randn(1, 16, 4, 4),
        torch.ones(1, 4, 4),
    )
    probability = model.semantic_probability(output)
    assert probability.shape == (1, 4, 4)
    assert torch.all((probability >= 0) & (probability <= 1))
