from casbl_jadce.plotting import figure_formats


def test_figure_format_options():
    assert figure_formats("none") == ()
    assert figure_formats("png") == ("png",)
    assert figure_formats("pdf") == ("pdf",)
    assert figure_formats("both") == ("png", "pdf")


def test_activity_s_distribution_uses_saved_task_data(tmp_path):
    import numpy as np
    from casbl_jadce.activity_model import ActivitySample
    from casbl_jadce.dataset import save_activity_samples
    from casbl_jadce.plotting import plot_activity_s_distribution

    def sample(active_count):
        a = np.zeros(6, dtype=bool); a[:active_count] = True
        return ActivitySample(
            a=a,
            device_locations=np.zeros((6, 2)),
            event_locations=np.zeros((1, 2)),
            activation_probabilities=np.full(6, active_count / 6),
        )

    out = tmp_path / "run"
    save_activity_samples(out / "activity" / "tuning_activity.npz", [sample(1), sample(2)])
    save_activity_samples(out / "activity" / "evaluation_activity.npz", [sample(2), sample(3)])
    cfg = {"run": {"output_dir": str(out)}, "system": {"S": 2}}
    paths = plot_activity_s_distribution(cfg, formats="png")
    assert len(paths) == 1
    assert paths[0].exists()


def test_activity_probability_grid_matches_event_model():
    import numpy as np
    from casbl_jadce.plotting import activity_probability_grid

    X, Y, P = activity_probability_grid(
        np.array([[0.0, 0.0]]), R=20.0, D=10.0, kappa=3.0, resolution=101
    )
    center = 50
    assert np.isclose(P[center, center], 1.0)
    # x=12 m is inside the BS cell but outside the event cutoff D=10 m.
    j = int(np.argmin(np.abs(X[center] - 12.0)))
    assert np.isclose(P[center, j], 0.0)
    # A corner is outside the circular BS cell and must be masked.
    assert bool(P.mask[0, 0])


def test_activity_probability_figure_can_be_saved(tmp_path):
    import numpy as np
    from casbl_jadce.activity_model import ActivitySample
    from casbl_jadce.plotting import plot_activity_sample

    sample = ActivitySample(
        a=np.array([True, False]),
        device_locations=np.array([[0.0, 0.0], [8.0, 0.0]]),
        event_locations=np.array([[0.0, 0.0]]),
        activation_probabilities=np.array([1.0, 0.1]),
    )
    paths = plot_activity_sample(
        sample, R=20.0, D=10.0, kappa=3.0,
        stem=tmp_path / "probability", formats="png",
        show_probability_field=True, probability_resolution=101,
    )
    assert len(paths) == 1
    assert paths[0].exists()

def test_probability_field_uses_vector_contours(tmp_path, monkeypatch):
    import matplotlib.axes
    import numpy as np
    from casbl_jadce.activity_model import ActivitySample
    from casbl_jadce.plotting import plot_activity_sample

    called = {"contourf": False}
    original_contourf = matplotlib.axes.Axes.contourf

    def spy_contourf(self, *args, **kwargs):
        called["contourf"] = True
        return original_contourf(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "contourf", spy_contourf)
    sample = ActivitySample(
        a=np.array([True, False]),
        device_locations=np.array([[0.0, 0.0], [8.0, 0.0]]),
        event_locations=np.array([[0.0, 0.0]]),
        activation_probabilities=np.array([1.0, 0.1]),
    )
    paths = plot_activity_sample(
        sample, R=20.0, D=10.0, kappa=3.0,
        stem=tmp_path / "vector_probability", formats="pdf",
        show_probability_field=True, probability_resolution=101,
        probability_levels=32,
    )
    assert called["contourf"]
    assert len(paths) == 1
    assert paths[0].exists()


def test_probability_contours_start_at_zero_and_colormap_zero_is_white(monkeypatch, tmp_path):
    import matplotlib.axes
    import numpy as np
    from casbl_jadce.activity_model import ActivitySample
    from casbl_jadce.plotting import plot_activity_sample

    captured = {}
    original = matplotlib.axes.Axes.contourf

    def spy(self, *args, **kwargs):
        captured["levels"] = np.asarray(kwargs["levels"], dtype=float)
        captured["cmap"] = kwargs["cmap"]
        captured["norm"] = kwargs["norm"]
        return original(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "contourf", spy)
    sample = ActivitySample(
        a=np.array([True, False]),
        device_locations=np.array([[0.0, 0.0], [5.0, 5.0]]),
        event_locations=np.array([[0.0, 0.0]]),
        activation_probabilities=np.array([1.0, 0.1]),
    )
    plot_activity_sample(
        sample, R=20.0, D=10.0, kappa=3.0,
        stem=tmp_path / "probability", formats="png",
        show_probability_field=True, probability_levels=16,
    )

    assert captured["levels"][0] == 0.0
    assert captured["levels"][-1] == 1.0
    assert captured["norm"].vmin == 0.0
    assert captured["norm"].vmax == 1.0
    assert np.allclose(captured["cmap"](0.0)[:3], [1.0, 1.0, 1.0])

