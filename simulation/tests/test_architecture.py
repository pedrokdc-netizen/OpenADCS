from pathlib import Path

import pytest

from openadcs_sim.architecture import CONNECTIONS, DIAGRAM_SIZE, MODULES, ModuleStatus


ROOT = Path(__file__).resolve().parents[2]
PLACEHOLDERS = [placeholder for module in MODULES for placeholder in module.placeholders]


def test_catalog_covers_required_modules_and_reports_actual_capabilities():
    modules = {module.id: module for module in MODULES}
    assert len(modules) == len(MODULES)
    assert {"orbit", "magnetic", "sun", "conditions", "dynamics",     "fine", "mag",
            "gyro", "horizon", "gps", "adcs", "obc", "actuators"} == set(modules)
    assert modules["gps"].status is ModuleStatus.IMPLEMENTED
    assert modules["gyro"].status is ModuleStatus.IMPLEMENTED
    for key in ("fine", "adcs", "conditions", "horizon"):
        assert modules[key].status is ModuleStatus.PARTIAL
    for key in ("obc", "actuators", "dynamics"):
        assert modules[key].status is ModuleStatus.PLACEHOLDER
    for module in MODULES:
        assert module.inputs and module.outputs and module.source_paths
        assert bool(module.placeholders) == (module.status is not ModuleStatus.IMPLEMENTED)
        assert all(ROOT.joinpath(*path.split("\\")).is_file() for path in module.source_paths)


@pytest.mark.parametrize("placeholder", PLACEHOLDERS, ids=lambda item: item.module_id)
def test_boilerplates_fail_explicitly_instead_of_returning_measurements(placeholder):
    with pytest.raises(NotImplementedError, match=f"{placeholder.module_id} is a PLACEHOLDER"):
        placeholder().evaluate(0, {})


def test_connections_have_unique_endpoints_and_bounded_orthogonal_routes():
    modules = {module.id: module for module in MODULES}
    assert len({connection.id for connection in CONNECTIONS}) == len(CONNECTIONS)
    for connection in CONNECTIONS:
        assert connection.source in modules and connection.target in modules
        assert connection.variables
        for point in (*connection.points, connection.label_position):
            assert 0 <= point[0] <= DIAGRAM_SIZE[0]
            assert 0 <= point[1] <= DIAGRAM_SIZE[1]
        for first, second in zip(connection.points, connection.points[1:]):
            assert (first[0] == second[0]) != (first[1] == second[1])
        for key, point in ((connection.source, connection.points[0]),
                           (connection.target, connection.points[-1])):
            left, top, right, bottom = modules[key].bounds
            assert (point[0] in (left, right) and top <= point[1] <= bottom) or (
                point[1] in (top, bottom) and left <= point[0] <= right
            )
        if modules[connection.source].status is ModuleStatus.PLACEHOLDER or (
            modules[connection.target].status is ModuleStatus.PLACEHOLDER
        ):
            assert connection.planned
    assert all(any(connection.source == module.id or connection.target == module.id
                   for connection in CONNECTIONS) for module in MODULES)


def test_reference_truth_is_not_presented_as_measured_attitude_or_live_control():
    connections = {connection.id: connection for connection in CONNECTIONS}
    assert "NOT a measurement" in connections["magnetic_reference"].variables
    assert connections["magnetic_reference"].planned
    assert "ECEF PVT" in connections["gps_adcs"].variables
    assert not connections["gps_adcs"].planned
    assert connections["adcs_actuators"].planned
    assert connections["actuators_dynamics"].planned
