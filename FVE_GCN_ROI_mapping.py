import numpy as np
import scipy.io
from nilearn import datasets, surface
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
import nibabel as nib
import pandas as pd
import numpy as np
from FVE_GCN_utils import *




partial_dat = False
partial_tsa_dat = False
icld_age_sex = False
if partial_dat:
    FVE_df_org = pd.read_csv("data/FVE_dat_partial.csv")
    SurfeView_surfaces = scipy.io.loadmat("data/SurfeView_surfaces.mat")
    non_surface_area_vars = ["nihtbx_cryst_uncorrected"]
    dat_type = "GCN_partial"
    dat_type2 = "partial"
elif partial_tsa_dat:
    FVE_df_org = pd.read_csv("data/FVE_dat_partial_tsa.csv")
    SurfeView_surfaces = scipy.io.loadmat("data/SurfeView_surfaces.mat")
    non_surface_area_vars = ["nihtbx_cryst_uncorrected"]
    dat_type = "GCN_partial_tsa"
    dat_type2 = "partial_tsa"
else:
    FVE_df_org = pd.read_csv("data/FVE_dat.csv")
    SurfeView_surfaces = scipy.io.loadmat("data/SurfeView_surfaces.mat")
    non_surface_area_vars = ["interview_age", "sex_2", "nihtbx_cryst_uncorrected"]
    dat_type2 = "regular"
    if icld_age_sex:
        dat_type = "GCN_cov"
    else:
        dat_type = "GCN"

# original mapping
SurfeView_surfaces = scipy.io.loadmat("data/SurfeView_surfaces.mat")
vertices_coords_orgs, faces_orgs = load_surface_mesh(SurfeView_surfaces)
vertices_coords_orgs_left = vertices_coords_orgs[:10242]
vertices_coords_orgs_right = vertices_coords_orgs[10242:]
print(vertices_coords_orgs.shape, faces_orgs.shape)

# load FreeSurfer 20K resolution
fsaverage = datasets.fetch_surf_fsaverage('fsaverage5')

coords_fs_left, faces_fs_left = surface.load_surf_mesh(fsaverage.pial_left)
coords_fs_right, faces_fs_right = surface.load_surf_mesh(fsaverage.pial_right)



vertices_coords_fs = np.vstack([coords_fs_left, coords_fs_right])
faces_fs = np.vstack([faces_fs_left, faces_fs_right+10242])

print(coords_fs_left.shape, coords_fs_right.shape)
print(vertices_coords_fs.shape, faces_fs.shape)


FVE_df_old_X = FVE_df_org.dropna()

left_cols = [c for c in FVE_df_old_X.columns if c.endswith('_l')]
right_cols = [c for c in FVE_df_old_X.columns if c.endswith('_r')]
FVE_df_fs = map_to_fs(FVE_df_old_X, coords_fs_left, coords_fs_right, vertices_coords_orgs_left, vertices_coords_orgs_right, non_surface_area_vars)




# Load the CIFTI-2 dense label file
img = nib.load("data_out/abcd_template_matching_combined_clusters_thresh0.61.dlabel.nii")

# Get the data (shape = (1, 91282))
data = img.get_fdata().astype(int).flatten()


# Get the label axis (axis 0)
label_axis = img.header.get_axis(0)
print(f"Label axis type: {type(label_axis)}")

# Handle nibabel version differences
label_table = label_axis.label[0]

print(f"Number of labels: {len(label_table)}")
print(f"Total vertices: {len(data)}")

# Each label entry is a tuple: (name, (R,G,B,A))
for key, value in label_table.items():
    label_name, rgba = value
    print(f"Label {key}: '{label_name}', RGBA={rgba}")


# Dictionary: ROI name -> vertex indices
roi_vertices = {}

for label_value, (label_name, rgba) in label_table.items():
    # find vertices belonging to this ROI
    indices = np.where(data == label_value)[0]
    roi_vertices[label_name] = indices
    print(f"{label_name}: {len(indices)} vertices")




from nilearn import plotting, surface
label_L = nib.load("data_out/abcd_labels.L.10k.label.gii")
label_R = nib.load("data_out/abcd_labels.R.10k.label.gii")


data_L = label_L.darrays[0].data
data_R = label_R.darrays[0].data

print(f"Left hemisphere vertices: {len(data_L)}")
print(f"Right hemisphere vertices: {len(data_R)}")

print(f"Unique labels (L): {np.unique(data_L)}")
print(f"Unique labels (R): {np.unique(data_R)}")


fsavg = surface.load_surf_mesh("data_out/fs_LR.10k.L.pial.surf.gii")


brain_mapping_mean_df = summarize_by_brain_region(df=FVE_df_fs, data_l=data_L, data_r=data_R)

if dat_type2 == "regular":
    brain_mapping_mean_df = pd.concat([brain_mapping_mean_df, FVE_df_org[['interview_age', 'sex_2', 'nihtbx_cryst_uncorrected']]], axis=1)
else:
    brain_mapping_mean_df = pd.concat([brain_mapping_mean_df, FVE_df_org[['nihtbx_cryst_uncorrected']]], axis=1) 


brain_mapping_mean_df.to_csv(f"/niddk-data-central/mae_hr/FVE/data_out/brain_mapping_mean_{dat_type2}.csv")


from sklearn.model_selection import train_test_split
train_valid, test = train_test_split(brain_mapping_mean_df, test_size=0.2, random_state=42)

train_valid.to_csv(f"/niddk-data-central/mae_hr/FVE/data_out/brain_mapping_mean_{dat_type2}_train.csv")
test.to_csv(f"/niddk-data-central/mae_hr/FVE/data_out/brain_mapping_mean_{dat_type2}_test.csv")




