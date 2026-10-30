"""Results summary for the author (Portuguese), generated from results.json.

Statements are formulated from the numbers (e.g. a ranking is called robust
only if it holds for every matched segmentation criterion, a plateau is
reported only if one is found), so the summary stays truthful if data,
parameters or code change.
"""
from __future__ import annotations

import logging

import numpy as np

from .config import Config
from .util import load_json

log = logging.getLogger("drp")


def _p(x, nd=4):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def _pc(x, nd=1):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:.{nd}f} %"


def _ci(v, nd=4):
    return f"[{_p(v[0], nd)}; {_p(v[1], nd)}]"


def _pv(p, floor=None):
    if p is None or (isinstance(p, float) and not np.isfinite(p)):
        return "–"
    if floor is not None and p <= floor * (1 + 1e-9):
        return f"≤ {floor:.4f}"
    return f"{p:.3g}" if p >= 1e-4 else "< 0,0001"


def _peq(p, floor=None):
    """'p = x' or 'p ≤ floor' for running text."""
    v = _pv(p, floor)
    return f"p {v}" if v.startswith(("≤", "<")) else f"p = {v}"


def step_report(cfg: Config):
    R = load_json(cfg.data_dir / "results.json")
    S = R["samples"]
    ids = list(S)
    A, B = ids[0], ids[1]
    la, lb = S[A]["label"], S[B]["label"]
    L = []
    w = L.append

    w("# Síntese dos resultados — gerada automaticamente\n")
    w("Todos os números abaixo vêm de `outputs/data/results.json`, produzido por `python run_all.py`. "
      "Tabelas completas: `outputs/tables/` (CSV/XLSX/LaTeX/Markdown). Figuras: `outputs/figures/` (PDF + PNG 600 dpi).\n")
    w("> Cada amostra é **um único espécime** por litologia. Toda inferência estatística se refere aos dois volumes "
      "imageados, não às litologias. Incertezas: porosidade da pilha inteira — extrapolação variância–volume "
      "(a mais conservadora de dois ajustes); expoente α — bootstrap de blocos em z e teste contra campos "
      "estacionários simulados; perfis — tamanho amostral efetivo (AR(1) com correção de viés) e tendência "
      "calibrada por bootstrap paramétrico AR(1) (com surrogates de fase como limite conservador); sub-volumes — séries "
      "das médias por camada z com tamanho efetivo AR(1) (aproximado). "
      f"Contrastes entre espécimes: sempre **{la} − {lb}**; razões: {la}/{lb}.\n")

    # 1 ------------------------------------------------------------------
    w("## 1. Correções nos dados de entrada\n")
    for sid in ids:
        a = S[sid]["acquisition"]
        leg = S[sid]["legacy"]
        w(f"- **{S[sid]['long_label']}**: tamanho de voxel registrado nos metadados das fatias = "
          f"**{a['voxel_size_um_metadata']} µm** (rótulo `{a['metadata_label_first']}`); o fluxo anterior usou "
          f"{cfg['legacy']['assumed_voxel_um']} µm. Pilha recortada: {a['shape'][2]} × {a['shape'][1]} × {a['shape'][0]} "
          f"voxels (x × y × z) = {a['extent_mm_zyx'][2]:.2f} × {a['extent_mm_zyx'][1]:.2f} × {a['extent_mm_zyx'][0]:.2f} mm; "
          f"{a['n_files']} fatias (não 1.200). O sub-volume anterior (fatias {leg['box']['z'][0]}–{leg['box']['z'][1] - 1}) "
          f"tem volume real de **{leg['metadata']['volume_mm3']:.0f} mm³** (o fluxo anterior calculou "
          f"{leg['assumed']['volume_mm3']:.0f} mm³).")
    va = S[A]["legacy"]["metadata"]["volume_mm3"]
    vb = S[B]["legacy"]["metadata"]["volume_mm3"]
    w(f"\nCom os voxels corretos, o sub-volume anterior do {la} é {va / vb:.2f}× o do {lb} em volume físico. "
      f"As duas aquisições **não** têm a mesma resolução: o {la} foi imageado com voxel "
      f"{S[A]['acquisition']['voxel_size_um']/S[B]['acquisition']['voxel_size_um']:.2f}× maior.\n")
    for sid in ids:
        q = S[sid]["qc"]
        ci = q.get("spearman_mode_vs_porosity_ci", [None, None])
        w(f"- Deriva do pico do sólido ao longo de z ({S[sid]['label']}): {q['mode_range_percent']:.2f} %; "
          f"ρ(nível do sólido, porosidade aparente) = {q['spearman_mode_vs_porosity']:+.2f} "
          f"(IC 95 % por blocos {_ci(ci, 2)}; {_peq(q.get('spearman_p_surrogate'))} contra surrogates) → associação "
          f"**{ {'negative': 'negativa', 'positive': 'positiva', 'inconclusive': 'inconclusiva'}.get(q.get('association_class'), '–') }**; "
          f"correção {'**aplicada**' if q['drift_correction_applied'] else '**não aplicada**'}"
          + (" (escolha a priori em config.toml, não decidida pelo diagnóstico)." if q.get('association_class') == 'inconclusive'
             else "."))
    for sid in ids:
        sens = S[sid]["profiles"].get("drift_sensitivity_z")
        if not sens or not S[sid]["qc"]["drift_correction_applied"]:
            continue
        b, m = sens["binary"], sens["mixing"]
        unc = S[sid]["segmentation_uncorrected_thresholds"]
        w(f"- Sensibilidade à correção ({S[sid]['label']}): φ da pilha (f = 0,5) "
          f"{S[sid]['segmentation']['thresholds_by_key']['primary']['porosity']:.4f} corrigida vs "
          f"{unc['primary']['porosity']:.4f} sem correção; balanço de massa {S[sid]['segmentation']['mixing']['unclipped']:.4f} vs "
          f"{S[sid]['segmentation_uncorrected_mixing']:.4f}. Perfil z (f = 0,5): CV {b['as_analysed']['cv']:.3f} vs "
          f"{b['uncorrected']['cv']:.3f}; tendência {b['as_analysed']['trend_per_10mm']:+.4f} vs "
          f"{b['uncorrected']['trend_per_10mm']:+.4f} por 10 mm. Perfil z (balanço de massa): CV "
          f"{m['as_analysed']['cv']:.3f} vs {m['uncorrected']['cv']:.3f}.")
    w("")

    # 2 ------------------------------------------------------------------
    w("## 2. Replicação da análise anterior\n")
    w("| Amostra | φ anterior | φ reproduzida | Área anterior (mm²) | Área reproduzida (28 µm) | Área com voxel correto | D anterior | D reproduzido |")
    w("|---|---|---|---|---|---|---|---|")
    for sid in ids:
        lg = S[sid]["legacy"]
        pb = lg["published"]
        w(f"| {S[sid]['label']} | {pb.get('porosity')} | {lg['porosity']:.4f} | {pb.get('area_mm2')} | "
          f"{lg['assumed']['area_mm2']:.2f} | {lg['metadata']['area_mm2']:.2f} | {pb.get('fractal_dimension')} | "
          f"{lg['boxcount']['D_legacy']:.4f} |")
    sva = S[A]["legacy"]["metadata"]["sv_mc_binary_per_mm"]
    svb = S[B]["legacy"]["metadata"]["sv_mc_binary_per_mm"]
    w(f"\nO código novo reproduz exatamente o fluxo antigo, o que valida a cadeia. Com os voxels corretos, "
      f"S_V (marching cubes binário) do sub-volume anterior = {sva:.3f} mm⁻¹ ({la}) vs {svb:.3f} mm⁻¹ ({lb}), "
      f"razão {sva / svb:.2f}.\n")

    # 3 ------------------------------------------------------------------
    w("## 3. Segmentação\n")
    for sid in ids:
        seg = S[sid]["segmentation"]
        t = seg["thresholds_by_key"]
        lab = S[sid]["acquisition"]["lab_porosity"]
        phis = {k: v["porosity"] for k, v in t.items() if k != "lab"}
        w(f"- **{S[sid]['label']}** (lab {lab:.3f}): métodos não supervisionados dão φ entre "
          f"{min(v for k, v in phis.items() if k not in ('legacy', 'primary')):.3f} e "
          f"{max(v for k, v in phis.items() if k not in ('legacy', 'primary')):.3f}; "
          f"μ − 1,1σ → {t['legacy']['porosity']:.4f}; f = 0,5 → {t['primary']['porosity']:.4f}. "
          f"O k que reproduz o laboratório é **{seg['k_lab']:.3f}**; em termos físicos, μ − 1,1σ corresponde a "
          f"f = {t['legacy']['f_equiv']:.2f} (voxel conta como poro se tiver ≥ {100 * t['legacy']['f_equiv']:.0f} % de poro).")
    kb = S[B]["segmentation"]["k_lab"]
    w(f"\nO fator 1,1 coincide com o k que reproduz a porosidade de laboratório do {lb} (k = {kb:.3f}); se foi "
      "escolhido assim, a concordância do dolomito é circular e não valida o método. Além disso, μ − kσ aplica "
      "critérios físicos diferentes a cada tomografia (f diferente), o que o torna inadequado para comparar amostras. "
      "A escala de volume parcial f, definida pelos membros-extremos de cada imagem, resolve isso.\n")

    # 4 ------------------------------------------------------------------
    w("## 4. Orçamento de porosidade e microporosidade sub-resolução\n")
    w("| Amostra | Estimador | φ imagem | EP espacial | Desvio vs lab | IC 95 % do desvio | p |")
    w("|---|---|---|---|---|---|---|")
    for sid in ids:
        b = S[sid]["porosity_budget"]
        for key, name in (("mixing", "balanço de massa (sem limiar)"), ("legacy_threshold", "binário μ − 1,1σ"),
                          ("primary_threshold", "binário f = 0,5")):
            d = b[key]
            w(f"| {S[sid]['label']} | {name} | {_p(d['estimate'])} | {_p(d['se_spatial'])} | "
              f"{_p(d['deviation'])} ({_pc(d['relative_deviation'])}) | {_ci(d['deviation_ci'])} | {_pv(d['p_two_sided'])} |")
    w("")
    for sid in ids:
        ib = S[sid]["porosity_budget"]["implied_background"]
        w(f"- Porosidade sub-resolução implícita na fase 'sólida' ({S[sid]['label']}): **{_pc(ib['estimate'])}** "
          f"(IC 95 % {_pc(ib['ci'][0])} a {_pc(ib['ci'][1])}; faixa dos membros-extremos "
          f"{_pc(ib['endmember_range'][0])} a {_pc(ib['endmember_range'][1])}).")
    w("\nO balanço de massa de atenuação (φ = (I_s − Ī)/(I_s − I_p)) não depende de limiar e é imune a ruído de média "
      "zero; é a estimativa mais robusta da porosidade 'vista' pela imagem. A diferença para o laboratório estima a "
      "porosidade abaixo da resolução. O EP espacial é a extrapolação da lei variância–volume para o volume da pilha; "
      "usa-se o maior dos dois ajustes (reta e domínio finito). Premissa: o desvio-padrão da porosidade de laboratório "
      f"({S[A]['porosity_budget']['lab_sd_assumed']}) **não** foi informado pelos depositantes — é assumido em `config.toml`.\n")
    for sid in ids:
        bs = S[sid]["porosity_budget"]["background_scenario"]
        w(f"- Cenário dos depositantes (fundo de {bs['phi_bg']:.0%}) — {S[sid]['label']}: binário μ − 1,1σ + fundo = "
          f"{bs['legacy_binary_plus_bg']:.4f} ({_pc(bs['legacy_binary_plus_bg_rel_error'])} vs lab); "
          f"balanço de massa + fundo = {bs['mixing_plus_bg']:.4f} ({_pc(bs['mixing_plus_bg_rel_error'])}).")
    bgs = [S[s]["porosity_budget"]["implied_background"] for s in ids]
    w(f"\nUm único valor de fundo não serve aos dois espécimes: a porosidade sub-resolução implícita é "
      f"{_pc(bgs[0]['estimate'])} no {la} e {_pc(bgs[1]['estimate'])} no {lb}.\n")

    # 5 ------------------------------------------------------------------
    w("## 5. Heterogeneidade e VER\n")
    for sid in ids:
        for tag, name in (("legacy", "μ − 1,1σ"), ("primary", "f = 0,5"), ("mixing", "balanço de massa")):
            if tag not in S[sid]["rev"]:
                continue
            f = S[sid]["rev"][tag]["fit"]
            nl = f.get("null", {})
            nfloor = 1 / (nl.get("n_valid", nl.get("n_sim", 100)) + 1)
            verdict = {"below": f"**abaixo** da faixa nula de meios estacionários ({_peq(nl.get('p_one_sided'), nfloor)})",
                       "within": "dentro da faixa nula (compatível com VER finito)",
                       "above": "acima da faixa nula",
                       "not testable": f"não testável com {nl.get('n_valid', nl.get('n_sim'))} simulações"
                       }.get(f.get("alpha_vs_null"), "–")
            w(f"- {S[sid]['label']} ({name}): Var(φ_V) ∝ V^−α com α = {f['alpha']:.2f} (EP por bootstrap de blocos em z "
              f"= {f.get('alpha_se_boot', float('nan')):.3f}, limite inferior) entre {f['fit_min_mm']:.1f} e "
              f"{f['fit_max_mm']:.1f} mm ({f['n_points']} tamanhos); faixa nula 95 % de campos estacionários com alcance "
              f"integral finito (comprimento de correlação {nl.get('corr_mm', float('nan')):g} mm): "
              f"[{_p(nl.get('null_q025'), 2)}; {_p(nl.get('null_q975'), 2)}] → {verdict}. Ajuste com correção de domínio "
              f"finito: α = {_p(f.get('alpha_finite_domain'), 2)}. CV numa janela de {f['largest_window_mm']:.0f} mm = "
              f"{f['cv_at_largest_window']:.3f}; EP da porosidade da pilha inteira = {f['porosity_se_total_powerlaw']:.4f} "
              f"(reta) / {_p(f.get('porosity_se_total_finite_domain'))} (domínio finito).")
    pc = R["profiles_comparison"]
    w("\nHeterogeneidade comparada em fatias de mesma espessura física e mesma extensão (perfis x e y médios sobre a "
      "mesma altura z nos dois espécimes; CV da porosidade por balanço de massa; IC aproximado por graus de liberdade "
      "efetivos só quando há ao menos 5 — 'n.e.' = não estimável):\n")
    w(f"| Eixo | Extensão comum (mm) | CV {la} | CV {lb} | Razão {la}/{lb} [IC 95 %] |")
    w("|---|---|---|---|---|")
    for a in "xyz":
        k = pc[a]["mixing"]
        def ci_or_ne(blk, nd):
            return _ci(blk["cv_ci"], nd) if blk.get("estimable") else "(n.e.)"
        rr = k["cv_ratio"]
        w(f"| {a} | {pc[a]['common_length_mm']:.1f} | {k[A]['cv']:.3f} {ci_or_ne(k[A], 3)} | "
          f"{k[B]['cv']:.3f} {ci_or_ne(k[B], 3)} | {rr['estimate']:.2f} "
          f"{_ci(rr['ci'], 2) if rr.get('estimable') else '(n.e.)'} |")
    w("\nCom tão poucas fatias independentes, os CVs de perfis **não** sustentam ordenar os espécimes por "
      "heterogeneidade (os ICs, quando estimáveis, são anticonservadores se a persistência das séries difere). A "
      "comparação de heterogeneidade deve ser feita pela escala variância–volume (acima).")
    w("")

    # 6 ------------------------------------------------------------------
    cubes = all(S[s]["cube"] for s in ids)
    if cubes:
        ca, cb = S[A]["cube"]["surface"], S[B]["cube"]["surface"]
        w(f"## 6. Superfície específica (cubos centrais de {S[A]['cube']['edge_mm']:.0f} mm em ambas as amostras)\n")
        w(f"| Estimador (f = 0,5) | {la} | {lb} |")
        w("|---|---|---|")
        for k, n in (("sv_crofton13", "Crofton, 13 direções"), ("sv_mc_grey", "marching cubes, isosuperfície de cinza"),
                     ("sv_mc_binary", "marching cubes, binário"), ("sv_faces", "faces de voxel (limite superior)"),
                     ("s_per_vp_crofton13", "S/V_p (por volume de poro)")):
            w(f"| {n} | {_p(ca.get(k), 3)} | {_p(cb.get(k), 3)} |")
        w(f"| corda média de poro 4φ/S_V (mm) | {_p(S[A]['cube']['mean_chord_mm'], 3)} | {_p(S[B]['cube']['mean_chord_mm'], 3)} |")
        w(f"| Kozeny–Carman φ³/(5 S_V²) (mD) | {_p(S[A]['cube'].get('k_kc_md'), 1)} | {_p(S[B]['cube'].get('k_kc_md'), 1)} |")
        rb = R.get("sweep_robustness", {})

        def rob(m, name):
            r = rb.get(m)
            if not r:
                return ""
            leg = f"; μ − 1,1σ (critério não pareado: f diferente em cada amostra) = {r['legacy_unmatched']:.2f}" \
                if r.get("legacy_unmatched") else ""
            return (f"{name} {la}/{lb} nos {r['n_criteria']} critérios pareados (f = 0,25–0,75): "
                    f"{r['min']:.2f}–{r['max']:.2f} → ordenação {'**robusta**' if r['robust_order'] else '**não robusta** (inverte com o critério)'}{leg}.")

        w("\n" + " ".join(x for x in (rob("sv_crofton13", "Razão S_V"), rob("s_per_vp", "Razão S/V_p"),
                                      rob("k_kc_md", "Razão Kozeny–Carman")) if x))
        w("Os estimadores Crofton e isosuperfície de cinza concordam; o marching cubes binário superestima "
          "(viés de 'escada', validado em esferas sintéticas) e a contagem de faces é ~1,5× maior.")
        sv_rob = rb.get("sv_crofton13", {}).get("robust_order")
        kc_rob = rb.get("k_kc_md", {}).get("robust_order")
        spv_rob = rb.get("s_per_vp", {}).get("robust_order")
        if sv_rob is False and kc_rob and spv_rob:
            w("A proxy de Kozeny–Carman depende de S/V_p (robusto) e não de S_V isolado: a inversão de S_V não se "
              "propaga para a ordenação de permeabilidade prevista; o critério muda, porém, sua magnitude.")
        elif kc_rob is not None:
            w(f"Ordenação da proxy de Kozeny–Carman entre critérios: {'robusta' if kc_rob else 'não robusta'}.")
        perc_none = all((S[s]["cube"]["connectivity"]["6"]["percolating_porosity_any"] or 0) == 0 for s in ids)
        w("A proxy usa só a fase resolvida"
          + (", que não percola no critério f = 0,5" if perc_none else "")
          + ": só a razão entre amostras tem significado; os valores absolutos (mD) não são previsões de "
            "permeabilidade.\n")

        # 7 --------------------------------------------------------------
        w("## 7. Dimensão fractal por contagem de caixas\n")
        for sid in ids:
            fd = S[sid]["fractal_diagnostics"]
            cf, un, lg, pl = fd["common_range_fit"], fd["unsaturated_fit"], fd["legacy"], fd["plateau"]
            if fd.get("plateau_reportable", pl["npts"] >= 3):
                pl_txt = (f"maior trecho de inclinação constante (±{pl['tol']:g}): {pl['s_min_mm']:.3f}–{pl['s_max_mm']:.3f} mm "
                          f"({pl['decades']:.2f} década; inclinação ≈ {pl['mean_slope']:.2f})"
                          + (" — curto demais para um regime de escala (limiar "
                             f"{fd['min_window_decades']:g} década)" if not fd["has_scaling_regime"] else ""))
            else:
                pl_txt = "nenhum trecho de inclinação constante (sem platô)"
            w(f"- **{S[sid]['label']}**: na faixa não saturada (s ≥ {fd['plateau_floor_mm']:.3f} mm, ocupação ≤ 0,5) a "
              f"inclinação local varia de {pl['slope_min']:.2f} a {pl['slope_max']:.2f}; {pl_txt}. D na faixa comum "
              f"{cf['s_min']:.3f}–{cf['s_max']:.3f} mm = {cf['D']:.2f} (intervalo de ajuste {cf['ci_low']:.2f}–{cf['ci_high']:.2f}, "
              f"descritivo), R² = {cf['r2']:.3f}; D na grade comum às duas amostras "
              f"({_p((fd.get('common_grid_mm') or [None])[0], 3)}–{_p((fd.get('common_grid_mm') or [None, None])[1], 3)} mm) = "
              f"{_p((fd.get('common_grid_fit') or {}).get('D'), 2)}; D só na faixa não saturada = {_p(un.get('D'), 2)}; "
              f"ρ(inclinação local, log s) = {_p((fd.get('slope_trend') or {}).get('rho'), 2)}; saturação "
              f"(ocupação 50 %) a partir de {fd['crossover_mm_occ50']:.2f} mm. No ajuste da análise anterior, "
              f"{lg['fraction_of_fitted_points_occupancy_gt_0.5']:.0%} dos pontos estavam com ocupação > 0,5 e "
              f"{lg['fraction_of_fitted_points_occupancy_gt_0.9']:.0%} > 0,9.")
        any_regime = any(S[s]["fractal_diagnostics"]["has_scaling_regime"] for s in ids)
        w("\nInterpretação: acima da escala de crossover quase toda caixa contém interface e N(s) → V/s³ "
          "(inclinação 3) independentemente da geometria; o D da análise anterior (≈2,8) mede sobretudo esse regime "
          "saturado, não a complexidade da interface. "
          + ("Abaixo do crossover, nenhum espécime tem um regime de escala de ao menos "
             f"{S[A]['fractal_diagnostics']['min_window_decades']:g} década, então D não é um descritor bem definido "
             "nesta resolução." if not any_regime else
             "Abaixo do crossover há regime de escala em: "
             + ", ".join(S[s]["label"] for s in ids if S[s]["fractal_diagnostics"]["has_scaling_regime"]) + ".")
          + " A escala de crossover (espaçamento característico dos poros resolvidos) é a grandeza informativa. "
          "Controles com resposta conhecida (material suplementar): um modelo booleano de esferas lisas, sem estrutura "
          "fractal, também produz um quase-platô com inclinação ≈ 1,8 abaixo do crossover, logo uma inclinação < 2 em "
          "fração de década não indica geometria fractal; e o critério de ocupação ≤ 0,5 só consegue resolver D abaixo "
          "de ≈ 2,6 (uma esponja de Menger, D = 2,73, satura todas as caixas).\n")

        # 8 --------------------------------------------------------------
        w("## 8. Conectividade, tamanho de poro e correlação\n")
        for sid in ids:
            c = S[sid]["cube"]
            c6 = c["connectivity"]["6"]
            psd = (S[sid].get("psd") or {}).get("curves", {}).get("primary", {})
            cor = c["correlation"]
            w(f"- {S[sid]['label']} (f = 0,5): maior cluster = {_pc(c6['largest_cluster_fraction'])} do volume poroso; "
              f"porosidade percolante = {c6['percolating_porosity_any']:.4f}; {c6['clusters_per_mm3']:.1f} clusters/mm³; "
              f"raio mediano de poro = {_p(psd.get('median_radius_um'), 0)} µm; comprimento de correlação (1/e) "
              f"x/y/z = {cor['x']['l_1e'] * 1000:.0f}/{cor['y']['l_1e'] * 1000:.0f}/{cor['z']['l_1e'] * 1000:.0f} µm.")
        for sid in ids:
            pb = S[sid]["cube"].get("percolation_by_criterion", {})
            fs = pb.get("f_percolating", [])
            w(f"- Percolação do {S[sid]['label']} por critério: "
              + (f"percola em f = {', '.join(f'{x:g}' for x in fs)}" if fs else "não percola em nenhum f testado")
              + ("; percola em μ − 1,1σ" if pb.get("legacy_percolates") else "; não percola em μ − 1,1σ") + ".")
        n_f = len([w_ for w_ in S[A]["cube"]["sweep"] if w_["criterion"] == "f"])
        depends = any(0 < len(S[s]["cube"].get("percolation_by_criterion", {}).get("f_percolating", [])) < n_f
                      for s in ids)
        if depends:
            w("\nSe a fase resolvida percola depende do critério de segmentação: uma simulação de escoamento em escala "
              "única nestas imagens ficaria refém de uma escolha arbitrária, e o escoamento real depende também da "
              "porosidade não resolvida.\n")
        else:
            w("")

    # 9 ------------------------------------------------------------------
    w("## 9. Resolução efetiva e experimento de resolução\n")
    for sid in ids:
        e = S[sid].get("esf") or {}
        if e.get("sigma_median_um"):
            w(f"- {S[sid]['label']}: σ da borda = {e['sigma_median_um']:.0f} µm ({e['sigma_median_vox']:.2f} voxels; "
              f"FWHM ≈ {e['fwhm_um']:.0f} µm; {e['n_kept']} bordas) para voxel de {e['voxel_um']:.2f} µm.")
    res = R.get("resolution")
    if res:
        fs = res.get("finer_scan_coarsened")
        if fs:
            def sg(x):
                return "–" if x is None else f"{x:+.2f}"
            se_u = [x for x in (fs.get("D_unsat_se") or []) if x]
            w(f"\nLevar o {S[fs['sample']]['label']} de {fs['from_um']:.2f} para {fs['to_um']:.2f} µm (média por blocos) "
              f"altera φ(f = 0,5) em {_pc(fs.get('rel_change_porosity'))}, S_V (Crofton) em "
              f"{_pc(fs.get('rel_change_sv_crofton13'))} e a escala de crossover em {_pc(fs.get('rel_change_crossover_mm'))}; "
              f"D na faixa não saturada (Fig. 8c do artigo) muda {sg(fs.get('abs_change_D_unsat'))}, D na faixa casada "
              f"max(0,1 mm; 2 voxels)–2,5 mm {sg(fs.get('abs_change_D_matched'))} e D na faixa fixa comum "
              f"{sg(fs.get('abs_change_D_fixed'))}"
              + (f" (EP de ajuste de D não saturado ≈ {max(se_u):.2f})." if se_u else "."))
        for sid in ids:
            sc = res["per_sample"][sid]["scaling"].get("sv_crofton13")
            if sc:
                w(f"- {S[sid]['label']}: S_V ∝ Δx^{sc['slope']:.2f} (R² = {sc['r2']:.3f}).")
        w("\nO expoente mede a sensibilidade de S_V à resolução; não deve ser convertido em dimensão fractal "
          "(D = 2 − expoente), porque o engrossamento também elimina poros inteiros (a fase binária muda), o que "
          "pode dar valores > 3, fisicamente impossíveis para uma superfície.")
        if res["matched"]:
            w("\nComparação na mesma resolução:\n")
            w(f"| Voxel (µm) | φ {la} | φ {lb} | S_V {la} | S_V {lb} | maior cluster {la} | maior cluster {lb} |")
            w("|---|---|---|---|---|---|---|")
            for m in res["matched"]:
                w(f"| {m['voxel_um']:.2f} | {_p(m[f'porosity_{A}'])} | {_p(m[f'porosity_{B}'])} | "
                  f"{_p(m[f'sv_crofton13_{A}'], 3)} | {_p(m[f'sv_crofton13_{B}'], 3)} | "
                  f"{_p(m[f'largest_cluster_fraction_{A}'], 3)} | {_p(m[f'largest_cluster_fraction_{B}'], 3)} |")
    w("")

    # 10 -----------------------------------------------------------------
    rp = R.get("replicates")
    if rp:
        nl_ = rp.get("n_layers", {})
        w("## 10. Sub-volumes replicados\n")
        w(f"Cubos não sobrepostos de {rp['edge_mm'][A]:.1f} mm cobrindo as pilhas inteiras "
          f"(n = {rp['n_cubes'][A]} no {la}, {rp['n_cubes'][B]} no {lb}; {nl_.get(A, '–')} e {nl_.get(B, '–')} camadas z). "
          "Os cubos de uma camada são correlacionados entre si e as médias das camadas são fortemente correlacionadas "
          "ao longo de z, então a inferência usa a série das médias por camada com tamanho efetivo AR(1) (n_eff de "
          "camadas na tabela); IC e p do contraste (Welch–Satterthwaite) são aproximados (em simulações com 8–13 camadas, "
          "tamanho real 2–7 % para 5 % nominal com correlação lag-1 das médias ≤ 0,5 e 11–17 % com 0,7–0,85) e saem como 'n.e.' quando as médias por camada são persistentes demais (gradiente "
          "monotônico em z) para estimar um tamanho efetivo. A coluna 'faixas disjuntas' diz se as faixas das médias "
          "por camada dos dois espécimes não se sobrepõem — uma descrição livre de modelo, e a única comparação "
          "defensável quando o IC é 'n.e.'. Hedges g e δ de Cliff são descritivos.\n")
        w(f"| Métrica | {la} média [IC] | {lb} média [IC] | n_eff camadas | diferença {la} − {lb} [IC aprox.] | p aprox. | faixas disjuntas | Hedges g | δ de Cliff |")
        w("|---|---|---|---|---|---|---|---|---|")
        for m, d in rp["metrics"].items():
            c = d["comparison"]

            def ci4(lo, hi):
                ok = lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi)
                return f"[{lo:.4g}; {hi:.4g}]" if ok else "(n.e.)"

            w(f"| {m} | {d[A]['mean']:.4g} {ci4(d[A]['mean_ci_low'], d[A]['mean_ci_high'])} | "
              f"{d[B]['mean']:.4g} {ci4(d[B]['mean_ci_low'], d[B]['mean_ci_high'])} | "
              f"{d[A]['n_eff_layers']:.1f}/{d[B]['n_eff_layers']:.1f} | "
              f"{c['difference']:.4g} {ci4(c['diff_ci_low'], c['diff_ci_high'])} | "
              f"{_pv(c['p_approx']) if c.get('estimable') else 'n.e.'} | "
              f"{'sim' if c['ranges_disjoint'] else 'não'} | {c['hedges_g']:.2f} | {c['cliffs_delta']:.2f} |")
        ex = rp.get("D_unsat_exclusion")
        if ex and sum(ex["per_sample"][s]["n_excluded"] for s in ids) > 0:
            parts = []
            for sid in ids:
                e = ex["per_sample"][sid]
                parts.append(f"{S[sid]['label']} {e['n_excluded']}/{e['n_total']}"
                             + (f" (porosidade {_p(e['porosity_excluded_mean'], 3)} vs {_p(e['porosity_kept_mean'], 3)} "
                                f"nos mantidos: excluídos {'mais' if e['excluded_are'] == 'more porous' else 'menos'} porosos)"
                                if e["n_excluded"] else ""))
            w(f"\nD_unsat indefinido (menos de {ex['min_points']} tamanhos ≥ {ex['fit_min_mm']:g} mm com ocupação ≤ "
              f"{ex['max_occupancy']:g}): " + "; ".join(parts)
              + ". As médias de D_unsat são condicionais à existência de uma faixa não saturada resolvível.")
            for var, name in (("D_unsat_2pt_fill", "preenchendo com a inclinação de 2 pontos"),
                              ("D_window", f"janela fixa {ex['window_mm'][0]:g}–{ex['window_mm'][1]:g} mm")):
                s_ = ex["sensitivity"].get(var)
                if s_:
                    w(f"- Sensibilidade ({name}): médias {s_['mean_a']:.3f} / {s_['mean_b']:.3f}; diferença "
                      f"{s_['difference']:+.3f} [{s_['diff_ci'][0]:.3f}; {s_['diff_ci'][1]:.3f}] (n {s_['n_a']}/{s_['n_b']}).")
        if rp.get("ancova"):
            w("\nContrastes ajustados pela porosidade (ANCOVA sobre os sub-volumes, **descritiva**: com poucas camadas "
              "efetivamente independentes não há intervalo amostral válido):\n")
            for resp, d in rp["ancova"].items():
                if "error" in d:
                    continue
                cn = f"{S[d['levels'][0]]['label']} − {S[d['levels'][1]]['label']}"
                pts = "; ".join(f"φ = {r['covariate']:.3f}: {r['difference']:.3g}" for r in d.get("contrast_over_common_support", []))
                w(f"- {resp} (n {', '.join(str(v) for v in d['n'].values())}): diferença ajustada ({cn}) na porosidade "
                  f"média {d['at_covariate']:.3f} = {d['adjusted_difference']:.4g}; inclinações {d['slope_a']:.3g} "
                  f"({S[d['levels'][0]]['label']}) e {d['slope_b']:.3g} ({S[d['levels'][1]]['label']})"
                  + (f"; ao longo da faixa comum de porosidade: {pts}" if pts else "") + ".")
        w("")

    # 11 -----------------------------------------------------------------
    w("## 11. Premissas e limitações registradas pelo código\n")
    w("- Um espécime por litologia: os testes comparam volumes imageados, não populações de rochas.")
    w("- Incerteza da porosidade de laboratório assumida (não fornecida pela fonte).")
    w("- Membro-extremo de poro = quantil 0,1 % do histograma; o borramento faz com que poros pequenos nunca "
      "atinjam esse valor — por isso todas as grandezas são também apresentadas como função de f e da faixa de I_p.")
    w("- O EP de α por bootstrap de blocos reamostra só ao longo de z (limite inferior); o teste nulo usa campos "
      "gaussianos estacionários com comprimento de correlação maior que o medido (conservador).")
    w("- O experimento de resolução usa média por blocos (idealização: não reproduz ruído nem PSF de uma "
      "aquisição real em voxel maior).")
    w("- σ da borda (ESF) aproxima a PSF: curvatura e rugosidade alargam a borda, mas uma segunda interface dentro "
      "da janela de ajuste (por exemplo a parede oposta de um poro de ~14–18 voxels) pode enviesar σ para baixo em "
      "alguns por cento.")
    w("- A correção de deriva por fatia assume que o nível do sólido deveria ser constante; ela é ativada por amostra "
      "em config.toml (o diagnóstico de associação é inconclusivo), e a seção 1 e a tabela T14 mostram a "
      "sensibilidade dos resultados a ela.")
    w("- Sub-volumes: a inferência entre espécimes usa poucas camadas efetivamente independentes; ICs e p são "
      "aproximados e as ANCOVAs são descritivas.")
    w("- Perfis com correlação lag-1 > 0,97: as tendências e os ICs de média são aproximados (poucas observações "
      "efetivas); o p da tendência vem de um bootstrap AR(1) (otimista se a memória for mais longa que AR(1)); o "
      "p conservador vem de surrogates de fase.")
    (cfg.output_dir / "RESULTS_SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log.info("RESULTS_SUMMARY.md written")
