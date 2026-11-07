"""Combined geometry semantics and real conformal export checks."""
from dataclasses import replace
import json
import numpy as np
import pytest
from tpmslab import Config, generate, save_model
from tpmslab.model import blend_weight, field_and_threshold, raw_field


@pytest.mark.parametrize('mode', ['sheet', 'solid_above', 'solid_below'])
def test_blend_end_regions_equal_original_families(mode):
    config = Config(combine='blend', mode=mode, m_cal=12, blend_calibration_steps=5)
    # Whole regions outside the transition, with different x/y and physical z.
    x = np.array([.27, 1.31, 3.42, 4.13])
    y = np.array([.63, 2.17, 4.39, 1.07])
    for z, family in [(np.full(4, .6), config.family), (np.full(4, 4.4), config.family_secondary)]:
        original = replace(config, combine='none', family=family)
        actual = raw_field(config, x, y, z)
        assert np.array_equal(actual, raw_field(original, x, y, z))


def test_same_family_blend_has_exact_original_field():
    c = Config(combine='blend', family_secondary='Gyroid', m_cal=12)
    x, y, z = np.meshgrid(np.linspace(0, 5, 7), np.linspace(0, 5, 6), np.linspace(0, 5, 9))
    assert np.array_equal(raw_field(c, x, y, z), raw_field(replace(c, combine='none'), x, y, z))


@pytest.mark.parametrize('profile', ['linear', 'cosine', 'smoothstep'])
@pytest.mark.parametrize('axis', ['x', 'y', 'z'])
def test_weight_endpoints_midpoint_and_bounds(axis, profile):
    c = Config(combine='blend', blend_axis=axis, blend_profile=profile)
    pos = [np.zeros(5) for _ in range(3)]
    pos['xyz'.index(axis)] = np.array([-1., 1.25, 2.5, 3.75, 6.])
    w = blend_weight(c, *pos)
    np.testing.assert_allclose(w, [0., 0., .5, 1., 1.], atol=1e-14)


def test_weight_remains_bounded_near_transition_endpoint():
    c = Config(combine='blend')
    z = np.linspace(3.749, 3.751, 1001)
    w = blend_weight(c, 0., 0., z)
    assert np.all((w >= 0.) & (w <= 1.))


def test_transition_is_continuous_at_clipped_bounds():
    c = Config(combine='blend', m_cal=12, blend_calibration_steps=5)
    for boundary in [1.25, 3.75]:
        z = np.array([boundary-1e-7, boundary, boundary+1e-7])
        f, q = field_and_threshold(c, .79, 1.23, z)
        assert np.ptp(f) < 2e-6 and np.ptp(q) < 2e-6


def test_midblend_quantile_calibrates_mixed_not_primary_field():
    c = Config(combine='blend', gradient='uniform', density_start=.4,
               phase_degrees=(0, 0, 0), m_cal=16, blend_calibration_steps=5)
    from tpmslab.model import blended_calibration, evaluate
    a = (np.arange(16)+.5)*2*np.pi/16
    angles = (a[:, None, None], a[None, :, None], a[None, None, :])
    f = .5*evaluate(c.expression, *angles) + .5*evaluate(c.expression_secondary, *angles)
    p, rows = blended_calibration(c.expression, c.expression_secondary, 'sheet', 16, 5)
    q = np.interp(.4, p, rows[2])
    # Discrete ties can bound the empirical fraction on either side of the target.
    assert np.mean(np.abs(f) < q) <= .4 + 1/4096
    assert np.mean(np.abs(f) <= q) >= .4 - 1/4096


@pytest.mark.parametrize('change', [
    {'combine':'bad'}, {'family_secondary':'bad'}, {'blend_axis':'bad'},
    {'blend_profile':'bad'}, {'blend_start':.8, 'blend_end':.2},
    {'blend_start':-.1}, {'blend_end':1.1}, {'blend_start':float('nan')},
    {'blend_start':True}, {'blend_calibration_steps':1}, {'blend_calibration_steps':True},
])
def test_invalid_combine_parameters_rejected(change):
    with pytest.raises(ValueError):
        Config.from_dict({**Config(combine='blend').to_dict(), **change})


def test_cancelling_custom_fields_rejected():
    c = Config(family='Custom', custom='sin(X)', family_secondary='Custom',
               custom_secondary='-sin(X)', combine='blend', m_cal=8, blend_calibration_steps=3)
    with pytest.raises(ValueError, match='constant at blend weight'):
        field_and_threshold(c, 1., 2., 2.5)


@pytest.mark.parametrize('mode', ['solid_above', 'solid_below'])
def test_combined_labyrinth_partition_across_multiple_cells(mode):
    from tpmslab.verification import audit_interface
    c = Config(combine='blend', family_secondary='Primitive Schwartz',
               mode=mode, cells=(2, 1, 1), blend_axis='x', axis='z',
               resolution=8, m_cal=12, blend_calibration_steps=5,
               domain_mode='solid_fluid')
    m = generate(c)
    p = m['points'][m['tetra']]
    v = np.einsum('ij,ij->i', p[:,1]-p[:,0], np.cross(p[:,2]-p[:,0],p[:,3]-p[:,0]))/6
    assert np.all(v > 0.)
    assert v.sum() == pytest.approx(125., abs=1e-9)
    assert audit_interface(m)['valid']


def test_combined_progress_and_report_remain_english():
    import re
    messages = []
    m = generate(Config(combine='blend', resolution=8, m_cal=12,
                        blend_calibration_steps=5, domain_mode='solid_fluid'),
                 progress=messages.append)
    assert messages
    text = json.dumps(m['report'], ensure_ascii=False) + '\n'.join(messages)
    assert not re.search(r'[\u4e00-\u9fff]', text)


@pytest.mark.parametrize('edge', ['linear', 'quadratic_root'])
def test_combined_complementary_partition_and_export(edge, tmp_path):
    c = Config(combine='blend', resolution=8, m_cal=12, blend_calibration_steps=5,
               domain_mode='solid_fluid', edge_interpolation=edge)
    m = generate(c)
    p = m['points'][m['tetra']]
    v = np.einsum('ij,ij->i', p[:,1]-p[:,0], np.cross(p[:,2]-p[:,0],p[:,3]-p[:,0]))/6
    assert np.all(v > 0)
    assert v.sum() == pytest.approx(125., abs=1e-9)
    assert set(m['phase_ids']) == {1, 2}
    assert len(m['interface']) > 0
    from tpmslab.verification import audit_interface
    assert audit_interface(m)['valid']
    folder = save_model(m, tmp_path/'combined')
    exported = json.loads((folder/'config.json').read_text())
    assert Config.from_dict(exported) == c
    assert 'CTETRA,' in (folder/'mesh.nas').read_text()
    from tpmslab.comsol import export_java
    export_java(folder, m['report'])
    assert (folder/'TPMSBuild.java').exists()
    assert m['report']['combined_design']['secondary_family'] == 'Diamond'


def test_solid_only_combined_mesh():
    m = generate(Config(combine='blend', resolution=8, m_cal=12, blend_calibration_steps=5))
    assert m['report']['watertight'] and 0 < m['report']['actual_density'] < 1


def test_http_combined_defaults_and_validation():
    from tpmslab.web import app, TOKEN
    client = app.test_client()
    assert client.get('/api/info').json['defaults']['combine'] == 'none'
    response = client.post('/api/generate', json={'combine':'blend','blend_end':-1},
                           headers={'X-TPMS-Token':TOKEN})
    assert response.status_code == 400
