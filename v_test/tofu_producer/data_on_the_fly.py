import scipy
import hydra
from omegaconf import DictConfig
import itertools
import random
import numpy as np
from mytools_0425 import Crystal
from pymatgen.core import Structure, Lattice
from matbench.bench import MatbenchBenchmark
from sklearn.preprocessing import PowerTransformer
import torch
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

folder = ""

def rotate_aug(s, permu_idx=0):
    lst = [0, 1, 2]
    permutations = list(itertools.permutations(lst))
    if permu_idx < 0 or permu_idx >= len(permutations):
        permu_idx = 0
    i, j, k = permutations[permu_idx][0], permutations[permu_idx][1], permutations[permu_idx][2]

    matrix = s.lattice.matrix
    species = s.species
    frac_coords = s.frac_coords
    new_matrix = np.array([matrix[i], matrix[j], matrix[k]])
    new_coords = []
    for coords in frac_coords:
        new_coords.append([coords[i], coords[j], coords[k]])
    new_coords = np.array(new_coords)

    new_structure = Structure(Lattice(new_matrix), species, new_coords)
    return new_structure

def random_walk_closest(structure, times):
    n = len(structure)
    idx_lst = list(range(n))
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = structure.get_distance(i, j)
            distance_matrix[i][j] = d
            distance_matrix[j][i] = d
    ret = []
    for _ in range(times):
        visited = []
        current_index = random.randint(0, n - 1)
        while len(visited) < n:
            while True:
                next_index = None
                d_min = 1e10
                visited.append(current_index)
                random.shuffle(idx_lst)
                for i in idx_lst:
                    if i not in visited:
                        d = distance_matrix[current_index][i]
                        if d < d_min:
                            d_min = d
                            next_index = i

                if next_index is not None:
                    current_index = next_index
                else:
                    break
            if len(visited) < n:
                print(visited)
                raise ValueError("random walk error")
        ret.append(visited)
    return ret

def gen_and_dump(name, fold, x_dataset, y_dataset, augmentation, norm, yeo_johnson_before_norm, path_precalculed_times):
    io_buffer = []
    io_cnt = 0
    if yeo_johnson_before_norm:
        # y_dataset, lambda_best_fit = scipy.stats.boxcox(y_dataset)
        # print(f'lambda in Box-Cox is: {lambda_best_fit}')
        pt = PowerTransformer(method='yeo-johnson', standardize=False)
        # print(type(y_dataset), y_dataset)
        y_dataset = pt.fit_transform(np.array(y_dataset).reshape(-1, 1))
        print(f"BoxCox Lambda with sklearn: {pt.lambdas_}")
    for i in range(len(x_dataset)):
        for j in range(augmentation):
            x = [(torch.tensor(c.img_latt, dtype=torch.float16),
                  torch.tensor(np.abs(c.img_atom_lst), dtype=torch.float16),
                  torch.tensor(np.angle(c.img_atom_lst), dtype=torch.float16),
                  torch.tensor(np.abs(c.img_atom), dtype=torch.float16),
                  torch.tensor(np.abs(c.img_atom_avg), dtype=torch.float16),
                  torch.tensor(np.angle(c.img_atom), dtype=torch.float16),
                  torch.tensor(np.angle(c.img_atom_avg), dtype=torch.float16))
                 for c in [Crystal(rotate_aug(x_dataset[i], j))]]
            # x = [(c.img_latt, np.real(c.img_atom_lst), np.imag(c.img_atom)) for c in [Crystal(rotate_aug(train_inputs[i], j))]]
            # if y_dataset[i] < 1e-5:
            #     continue
            if norm:
                y_mean = np.mean(y_dataset)
                y_std = np.std(y_dataset)
                y = [(y_dataset[i]-y_mean)/y_std]
            else:
                y_mean = None
                y_std = None
                y = [y_dataset[i]]
            y_train = torch.tensor(y, dtype=torch.float16)
            paths = []
            if path_precalculed_times > 0:
                paths = random_walk_closest(x_dataset[i], path_precalculed_times)
            data_dict_train = {
                "features_latt": x[0][0],
                "features_abs_lst": x[0][1],
                "features_angle_lst": x[0][2],
                "features_abs_sum": x[0][3],
                "features_abs_avg": x[0][4],
                "features_angle_sum": x[0][5],
                "features_angle_avg": x[0][6],
                "structure": x_dataset[i],
                "paths": paths,
                "labels": y_train}

            io_buffer.append(data_dict_train)
        if len(io_buffer) > 100 or i == len(x_dataset) - 1:
            for data_dict_train in io_buffer:
                torch.save(data_dict_train,
                           "{}/{}_{}_{}.pth".format(folder, name, fold, io_cnt))
                io_cnt += 1
            io_buffer.clear()
    print(f'{io_cnt} {name} file generated, mean {y_mean}, std {y_std}')

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(cfg)
    global folder
    folder = cfg.task.folder
    mb = MatbenchBenchmark(autoload=False, subset=[cfg.task.task_name])
    for task in mb.tasks:
        augmentation = cfg.task.augmentation  # change to 1 if no training rotate augmentation
        task.load()
        for fold in cfg.fold:
            # if fold in [0]:
            #     continue
            train_inputs, train_outputs = task.get_train_and_val_data(fold)
            random.seed(123)
            train_inputs, train_outputs = zip(
                *random.sample(list(zip(train_inputs, train_outputs)), len(train_inputs)))
    
            split_idx = int(cfg.task.split_ratio * len(train_inputs))
            vali_inputs, vali_outputs = train_inputs[split_idx:], train_outputs[split_idx:]
            train_inputs, train_outputs = train_inputs[:
                                                       split_idx], train_outputs[:split_idx]
            gen_and_dump("train", fold, train_inputs, train_outputs, augmentation, cfg.task.train_norm, cfg.task.yeo_johnson_before_norm, cfg.task.train_path_precalculated_times)
            gen_and_dump("vali", fold, vali_inputs, vali_outputs, augmentation, False, False, cfg.task.vali_path_precalculated_times)
    
            test_inputs, test_outputs = task.get_test_data(
                fold, include_target=True)
            gen_and_dump("test", fold, test_inputs, test_outputs, augmentation, False, False, cfg.task.test_path_precalculated_times)
            print('-' * 50)

if __name__ == "__main__":
    main()
