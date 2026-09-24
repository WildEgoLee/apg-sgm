import csv
import sys

filename = sys.argv[1] if len(sys.argv) > 1 else 'results/ablation_middlebury.csv'

with open(filename) as f:
    reader = csv.DictReader(f)
    rows = list(reader)

by_case = {}
for r in rows:
    by_case.setdefault(r['case'], {})[r['ablation_id']] = r

print(f"Total scenes: {len(by_case)}")
header = f"{'Scene':<14} {'Rec(vis)':<9} {'Rec(mat)':<9} {'P@1(vis)':<9} {'P@1(all)':<9} {'GridRec@1':<10} {'MAE(vis)':<9} {'PriorRed':<9} {'D Bad2':<8} {'E Bad2':<8} {'dBad2':<8} {'dEPE':<8}"
print(header)
print("-" * len(header))

rec_vis_list = []
rec_mat_list = []
prior_red_list = []
delta_bad2_list = []
delta_epe_list = []

tot_vis_eval = 0
tot_vis_in = 0
tot_mat_eval = 0
tot_mat_in = 0

tot_supp_gt_valid = 0
tot_supp_vis = 0
tot_supp_corr_05 = 0
tot_supp_corr_1 = 0
tot_supp_corr_2 = 0
tot_supp_vis_corr_05 = 0
tot_supp_vis_corr_1 = 0
tot_supp_vis_corr_2 = 0
tot_vis_gt_cells = 0
tot_corr_vis_cells = 0

supp_p05_list = []
supp_p1_all_list = []
supp_p1_vis_list = []
supp_p2_list = []
supp_mae_all_list = []
supp_mae_vis_list = []
supp_grid_rec_list = []

for case, abs_dict in by_case.items():
    d = abs_dict.get('D')
    e = abs_dict.get('E')
    if not e:
        continue
    rec_vis = float(e['recall_visible']) * 100
    rec_mat = float(e['recall_matchable']) * 100
    p_red = float(e['prior_incremental_reduction']) * 100
    d_bad2 = float(d['bad_2_0']) if d else 0.0
    e_bad2 = float(e['bad_2_0'])
    d_epe = float(d['epe']) if d else 0.0
    e_epe = float(e['epe'])
    del_bad2 = (e_bad2 - d_bad2) if d else 0.0
    del_epe = (e_epe - d_epe) if d else 0.0
    
    rec_vis_list.append(rec_vis)
    rec_mat_list.append(rec_mat)
    prior_red_list.append(p_red)
    if d:
        delta_bad2_list.append(del_bad2)
        delta_epe_list.append(del_epe)

    # Pixel counts for micro recall
    vis_eval = int(e.get('range_vis_eval_px', 0))
    vis_in = int(e.get('range_vis_in_px', 0))
    mat_eval = int(e.get('range_mat_eval_px', 0))
    mat_in = int(e.get('range_mat_in_px', 0))

    tot_vis_eval += vis_eval
    tot_vis_in += vis_in
    tot_mat_eval += mat_eval
    tot_mat_in += mat_in

    # Support raw counts
    tot_supp_gt_valid += int(e.get('support_gt_valid', 0))
    tot_supp_vis += int(e.get('support_visible', 0))
    tot_supp_corr_05 += int(e.get('support_correct_05', 0))
    tot_supp_corr_1 += int(e.get('support_correct_1', 0))
    tot_supp_corr_2 += int(e.get('support_correct_2', 0))
    tot_supp_vis_corr_05 += int(e.get('support_visible_correct_05', 0))
    tot_supp_vis_corr_1 += int(e.get('support_visible_correct_1', 0))
    tot_supp_vis_corr_2 += int(e.get('support_visible_correct_2', 0))
    tot_vis_gt_cells += int(e.get('support_visible_gt_cells', 0))
    tot_corr_vis_cells += int(e.get('support_correct_visible_cells', 0))

    # Support metrics
    p05 = float(e.get('support_pvis_05', e.get('support_p05', 0))) * 100
    p1_all = float(e.get('support_p1', 0)) * 100
    p1_vis = float(e.get('support_pvis_1', e.get('support_p1_vis', p1_all / 100.0))) * 100
    p2 = float(e.get('support_pvis_2', e.get('support_p2', 0))) * 100
    mae_all = float(e.get('support_mae', 0))
    mae_vis = float(e.get('support_mae_vis', mae_all))
    grid_rec = float(e.get('grid_recall_1', e.get('grid_coverage', 0))) * 100

    supp_p05_list.append(p05)
    supp_p1_all_list.append(p1_all)
    supp_p1_vis_list.append(p1_vis)
    supp_p2_list.append(p2)
    supp_mae_all_list.append(mae_all)
    supp_mae_vis_list.append(mae_vis)
    supp_grid_rec_list.append(grid_rec)
    
    d_bad2_str = f"{d_bad2:5.2f}%" if d else "  N/A "
    del_bad2_str = f"{del_bad2:+5.2f}%" if d else "  N/A "
    del_epe_str = f"{del_epe:+6.3f}" if d else "   N/A"
    print(f"{case:<14} {rec_vis:5.2f}%   {rec_mat:5.2f}%   {p1_vis:5.1f}%    {p1_all:5.1f}%    {grid_rec:5.1f}%     {mae_vis:5.2f}px   {p_red:5.1f}%    {d_bad2_str}   {e_bad2:5.2f}%   {del_bad2_str}  {del_epe_str}")

print("-" * len(header))
micro_vis = (tot_vis_in / tot_vis_eval * 100.0) if tot_vis_eval > 0 else 0.0
micro_mat = (tot_mat_in / tot_mat_eval * 100.0) if tot_mat_eval > 0 else 0.0

micro_supp_p05_vis = (tot_supp_vis_corr_05 / tot_supp_vis * 100.0) if tot_supp_vis > 0 else 0.0
micro_supp_p1_vis = (tot_supp_vis_corr_1 / tot_supp_vis * 100.0) if tot_supp_vis > 0 else 0.0
micro_supp_p2_vis = (tot_supp_vis_corr_2 / tot_supp_vis * 100.0) if tot_supp_vis > 0 else 0.0
micro_supp_grid_rec = (tot_corr_vis_cells / tot_vis_gt_cells * 100.0) if tot_vis_gt_cells > 0 else 0.0

print(f"Macro Visible Recall:     {sum(rec_vis_list)/len(rec_vis_list):.2f}%")
print(f"Micro Visible Recall:     {micro_vis:.2f}% (Total: {tot_vis_in}/{tot_vis_eval})")
print(f"Worst Scene Recall:       {min(rec_vis_list):.2f}%")
print(f"Macro Matchable Rec:      {sum(rec_mat_list)/len(rec_mat_list):.2f}% (Micro: {micro_mat:.2f}%)")
if tot_supp_vis > 0:
    print(f"Support Visible P@0.5:    Macro {sum(supp_p05_list)/len(supp_p05_list):.2f}% | Micro {micro_supp_p05_vis:.2f}% ({tot_supp_vis_corr_05}/{tot_supp_vis})")
    print(f"Support Visible P@1:      Macro {sum(supp_p1_vis_list)/len(supp_p1_vis_list):.2f}% | Micro {micro_supp_p1_vis:.2f}% ({tot_supp_vis_corr_1}/{tot_supp_vis})")
    print(f"Support Visible P@2:      Macro {sum(supp_p2_list)/len(supp_p2_list):.2f}% | Micro {micro_supp_p2_vis:.2f}% ({tot_supp_vis_corr_2}/{tot_supp_vis})")
    print(f"Support Grid Recall@1:    Macro {sum(supp_grid_rec_list)/len(supp_grid_rec_list):.2f}% | Micro {micro_supp_grid_rec:.2f}% ({tot_corr_vis_cells}/{tot_vis_gt_cells})")
else:
    print(f"Support Visible P@1:      {sum(supp_p1_vis_list)/len(supp_p1_vis_list):.2f}% (All P@1: {sum(supp_p1_all_list)/len(supp_p1_all_list):.2f}%)")
    print(f"Support Grid Recall@1:    {sum(supp_grid_rec_list)/len(supp_grid_rec_list):.2f}%")
print(f"Support Visible MAE:      {sum(supp_mae_vis_list)/len(supp_mae_vis_list):.3f} px (All MAE: {sum(supp_mae_all_list)/len(supp_mae_all_list):.3f} px)")
print(f"Prior Space Reduction:    {sum(prior_red_list)/len(prior_red_list):.2f}%")
if delta_bad2_list:
    print(f"Delta Bad-2 (D -> E):     {sum(delta_bad2_list)/len(delta_bad2_list):+.2f}% (Max increase: {max(delta_bad2_list):+.2f}%)")
    print(f"Delta EPE   (D -> E):     {sum(delta_epe_list)/len(delta_epe_list):+.3f} px")

if len(sys.argv) >= 3:
    packed_file = sys.argv[2]
    with open(packed_file) as pf:
        p_rows = list(csv.DictReader(pf))
    by_case_p = {}
    for r in p_rows:
        by_case_p.setdefault(r['case'], {})[r['ablation_id']] = r

    print("\n" + "=" * 120)
    print("DENSE vs PACKED BACKEND COMPARISON (Mode E: Prior + 4SGM)")
    print("=" * 120)
    cmp_hdr = f"{'Scene':<14} {'Dense CV Peak':<14} {'Packed CV Peak':<15} {'Peak Red%':<10} {'Dense SGM':<11} {'Pack SGM':<10} {'SGM Speed':<10} {'Dense Total':<12} {'Pack Total':<12} {'Pipe Speed':<10}"
    print(cmp_hdr)
    print("-" * len(cmp_hdr))

    peak_red_list = []
    sgm_spd_list = []
    tot_spd_list = []

    for case in by_case:
        if case not in by_case_p: continue
        d_e = by_case[case].get('E')
        p_e = by_case_p[case].get('E')
        if not d_e or not p_e: continue

        if d_e.get('backend') and d_e['backend'] != 'dense':
            raise ValueError(f"Expected dense backend in reference file, got '{d_e.get('backend')}'")
        if p_e.get('backend') and p_e['backend'] != 'packed':
            raise ValueError(f"Expected packed backend in target file, got '{p_e.get('backend')}'")

        must_equal = [
            'threads', 'effective_threads', 'warmup', 'repeat', 'build_type',
            'w', 'h', 'dmax', 'cost', 'cross', 'p2', 'prior', 'refine', 'paths'
        ]
        for key in must_equal:
            if key in d_e and key in p_e and d_e[key] != p_e[key]:
                raise ValueError(f"Incompatible benchmark parameter for {case} Mode E: {key} (Dense={d_e[key]} vs Packed={p_e[key]})")

        d_peak = float(d_e['peak_bytes']) / (1024 * 1024)
        p_peak = float(p_e['peak_bytes']) / (1024 * 1024)
        peak_red = (1.0 - p_peak / d_peak) * 100.0 if d_peak > 0 else 0.0

        d_sgm = float(d_e['time_sgm_ms'])
        p_sgm = float(p_e['time_sgm_ms'])
        sgm_spd = (d_sgm / p_sgm) if p_sgm > 0 else 1.0

        d_tot = float(d_e['time_total_ms'])
        p_tot = float(p_e['time_total_ms'])
        tot_spd = (d_tot / p_tot) if p_tot > 0 else 1.0

        peak_red_list.append(peak_red)
        sgm_spd_list.append(sgm_spd)
        tot_spd_list.append(tot_spd)

        print(f"{case:<14} {d_peak:10.2f} MB   {p_peak:11.2f} MB   {peak_red:6.2f}%    {d_sgm:7.2f} ms  {p_sgm:6.2f} ms  {sgm_spd:5.2f}x      {d_tot:8.2f} ms  {p_tot:8.2f} ms  {tot_spd:5.2f}x")

    print("-" * len(cmp_hdr))
    if peak_red_list:
        print(f"Mean Peak Cost-Volume Reduction: {sum(peak_red_list)/len(peak_red_list):.2f}% (min: {min(peak_red_list):.2f}%, max: {max(peak_red_list):.2f}%)")
        print(f"Mean SGM Speedup:               {sum(sgm_spd_list)/len(sgm_spd_list):.2f}x (min: {min(sgm_spd_list):.2f}x, max: {max(sgm_spd_list):.2f}x)")
        print(f"Mean Pipeline Speedup:          {sum(tot_spd_list)/len(tot_spd_list):.2f}x (min: {min(tot_spd_list):.2f}x, max: {max(tot_spd_list):.2f}x)")

