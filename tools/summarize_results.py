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
header = f"{'Scene':<15} {'Rec(vis)':<10} {'Rec(mat)':<10} {'Prior Red':<12} {'D Bad2':<10} {'E Bad2':<10} {'Delta Bad2':<12} {'D EPE':<10} {'E EPE':<10}"
print(header)
print("-" * len(header))

rec_vis_list = []
rec_mat_list = []
prior_red_list = []
delta_bad2_list = []
delta_epe_list = []

for case, abs_dict in by_case.items():
    d = abs_dict['D']
    e = abs_dict['E']
    rec_vis = float(e['recall_visible']) * 100
    rec_mat = float(e['recall_matchable']) * 100
    p_red = float(e['prior_incremental_reduction']) * 100
    d_bad2 = float(d['bad_2_0'])
    e_bad2 = float(e['bad_2_0'])
    d_epe = float(d['epe'])
    e_epe = float(e['epe'])
    del_bad2 = e_bad2 - d_bad2
    del_epe = e_epe - d_epe
    
    rec_vis_list.append(rec_vis)
    rec_mat_list.append(rec_mat)
    prior_red_list.append(p_red)
    delta_bad2_list.append(del_bad2)
    delta_epe_list.append(del_epe)
    
    print(f"{case:<15} {rec_vis:6.2f}%   {rec_mat:6.2f}%   {p_red:6.2f}%     {d_bad2:6.2f}%   {e_bad2:6.2f}%   {del_bad2:+6.2f}%     {d_epe:6.3f}   {e_epe:6.3f}")

print("-" * len(header))
print(f"Mean Rec(vis): {sum(rec_vis_list)/len(rec_vis_list):.2f}% (Min: {min(rec_vis_list):.2f}%)")
print(f"Mean Rec(matchable): {sum(rec_mat_list)/len(rec_mat_list):.2f}% (Min: {min(rec_mat_list):.2f}%)")
print(f"Mean Prior Reduction: {sum(prior_red_list)/len(prior_red_list):.2f}%")
print(f"Mean Delta Bad-2 (D->E): {sum(delta_bad2_list)/len(delta_bad2_list):+.2f}% (Max increase: {max(delta_bad2_list):+.2f}%)")
print(f"Mean Delta EPE (D->E): {sum(delta_epe_list)/len(delta_epe_list):+.3f} px")
