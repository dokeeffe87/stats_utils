import numpy as np

from stats_utils.ab_testing.conversion_rate import ConversionExperiment


def test_calc_delta_returns_absolute_minimum_detectable_effect():
    assert np.isclose(
        ConversionExperiment.calc_delta(
            baseline_conversion_rate=0.05,
            relative_minimum_detectable_effect_size=0.10,
        ),
        0.005,
    )


def test_calc_sigma_returns_standard_deviation_or_variance():
    assert ConversionExperiment.calc_sigma(0.05, return_square=True) == 0.0475
    assert np.isclose(ConversionExperiment.calc_sigma(0.05), np.sqrt(0.0475))


def test_calc_standard_error_can_use_baseline_or_sigma():
    experiment = ConversionExperiment()

    from_baseline = experiment.calc_standard_error(
        sample_size=100,
        baseline_conversion_rate=0.05,
    )
    from_sigma = experiment.calc_standard_error(
        sample_size=100,
        sigma=np.sqrt(0.0475),
    )

    assert np.isclose(from_baseline, from_sigma)
    assert np.isclose(from_baseline, np.sqrt(0.0475) * np.sqrt(2 / 100))


def test_calc_sample_size_preserves_existing_formula_output():
    experiment = ConversionExperiment()

    assert experiment.calc_sample_size(
        power=0.8,
        alpha=0.05,
        relative_minimum_detectable_effect_size=0.1,
        baseline_conversion_rate=0.05,
    ) == 29826
