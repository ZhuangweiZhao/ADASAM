from tools.profile_vaihingen_survival import aggregate


def test_efficiency_aggregate_groups_policies() -> None:
    rows = [
        {
            "policy": policy,
            "seed": seed,
            "mIoU_5": value,
            "Small_IoU": value,
            "FPS": value,
            "FLOPs": value,
            "peak_memory_allocated_MB": value,
            "executed_detail_projection_FLOPs_per_image": value,
            "P3_projected_positions": value,
            "P4_projected_positions": value,
        }
        for policy in ("random", "magnitude", "adaptive")
        for seed, value in ((42, 1.0), (123, 2.0), (456, 3.0))
    ]
    result = aggregate(rows)
    assert [item["policy"] for item in result] == ["random", "magnitude", "adaptive"]
    assert all(item["FPS_mean"] == 2.0 for item in result)
    assert all(item["FPS_std"] == 1.0 for item in result)
