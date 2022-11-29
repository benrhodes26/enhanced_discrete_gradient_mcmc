import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from samplers.regular_samplers import BinaryGWGSampler
from samplers.auxiliary_samplers import BGAVSampler
import sklearn.metrics
import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms as tr

from distributions.discrete import LatticeIsingModel, ERIsingModel, LatticePottsModel, QuadraticNeuralModel, BoltzmannMachine
from samplers.run_sample import get_sampler
from time import strftime, gmtime, time
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from utils import mmd
from utils.utils import numpify, torchify
from utils.my_usps import USPS

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')



def makedirs(dirname):
    """
    Make directory only if it's not already there.
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def l1(module):
    loss = 0.
    for p in module.parameters():
        loss += p.abs().sum()
    return loss


def graph_rocauc(true_G, est_G, neg_weights=0):
    aucroc = sklearn.metrics.roc_auc_score(y_true=(true_G > 0).flatten(), y_score=est_G.flatten())
    if neg_weights:
        neg_aucroc = sklearn.metrics.roc_auc_score(y_true=(true_G < 0).flatten(), y_score=-est_G.flatten())
        aucroc = (aucroc + neg_aucroc) / 2
    return aucroc


def diagnose_mmd(buffer, kmmd, method_save_dir, log_mmds, my_print, opt_log_mmd, trn_500):
    buffer_mmd = kmmd.compute_mmd(buffer[:500], trn_500)
    my_print(f"LOG MMD: {buffer_mmd.log10()}")
    log_mmds.append(buffer_mmd.log10().item())
    plt.clf()
    plt.plot(np.array(log_mmds), label="mmd")
    plt.plot([0, len(log_mmds)], [opt_log_mmd, opt_log_mmd], linestyle="--", c='k', label="target")
    plt.legend()
    plt.savefig("{}/log_mmds.png".format(method_save_dir))


def get_all_data(loader):
    all = []
    for x in loader:
        all.append(x[0])
    return torch.cat(all, dim=0).to(device)


def get_data(args):

    if args.data_file == "usps":
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        args.dim_sqrt = 16
        transform = tr.Compose([tr.Resize(args.dim_sqrt), tr.ToTensor(), lambda x: (x > .5).float().view(-1)])
        train_data = torchvision.datasets.USPS(root="./data", train=True, transform=transform, download=True)
        test_data = torchvision.datasets.USPS(root="./data", train=False, transform=transform, download=True)
        train_loader = DataLoader(train_data, args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_data, args.batch_size, shuffle=True, drop_last=True)
        sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
        plot = lambda p, x: torchvision.utils.save_image(x.view(x.size(0), 1, args.dim_sqrt, args.dim_sqrt),
                                                         p, normalize=True, nrow=sqrt(x.size(0)))
        encoder = None
        viz = None

    elif args.data_file == "histopathology":
        # ======================================================================================================================
        args.dim_sqrt = 10
        # start processing
        with open('./data/histopathology.pkl', 'rb') as f:
            data = pickle.load(f, encoding="latin1")

        def preprocess(x, thresh=0.5):
            x = torchify(np.array(x))
            x = T.CenterCrop(size=args.dim_sqrt)(x)
            x = torch.clip(x, 1. / 512., 1. - 1. / 512.)
            return numpify((x > thresh).float()).reshape(-1, args.dim_sqrt ** 2)

        med = np.median(np.array(data['training']))
        x_train = preprocess(data['training'], med)
        x_val = preprocess(data['validation'], med)
        x_test = preprocess(data['test'], med)

        # idle y's
        y_train = np.zeros((x_train.shape[0], 1))
        y_val = np.zeros((x_val.shape[0], 1))
        y_test = np.zeros((x_test.shape[0], 1))

        # pytorch data loader
        train = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train))
        train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)

        validation = TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
        val_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False)

        test = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
        test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=True)

        sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
        plot = lambda p, x: torchvision.utils.save_image(x.view(x.size(0), 1, args.dim_sqrt, args.dim_sqrt),
                                                         p, normalize=True, nrow=sqrt(x.size(0)))
        encoder = None
        viz = None

    elif args.data_file == "bsds300":

        args.dim_sqrt = 8

        def preprocess(x, thresh=0.5):
            x = (x > thresh).astype(np.float32).reshape(-1, 7 * 9)
            x = np.concatenate([x, np.random.randint(0, 2, size=(len(x), 1))], axis=-1)
            return x

        f = h5py.File('./data/BSDS300.hdf5', 'r')
        x_trn = f['train'][:100000]
        med = np.median(x_trn)
        x_train = preprocess(x_trn, med)
        x_val = preprocess(f['validation'][:10000], med)
        x_test = preprocess(f['test'][:10000], med)
        f.close()

        # idle y's
        y_train = np.zeros((x_train.shape[0], 1))
        y_val = np.zeros((x_val.shape[0], 1))
        y_test = np.zeros((x_test.shape[0], 1))

        # pytorch data loader
        train = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train))
        train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)

        validation = TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
        val_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False)

        test = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
        test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=True)

        sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
        plot = lambda p, x: torchvision.utils.save_image(x.view(x.size(0), 1, 8, 8), p, normalize=True, nrow=8)
        encoder = None
        viz = None

    elif args.data_file is not None:
        with open(args.data_file, 'rb') as f:
            if args.data_file.endswith(".npz"):
                x = np.load(f)["buffer"]
            else:
                x = pickle.load(f)
        x = torch.tensor(x).float()
        args.dim_sqrt = int(x.shape[1] ** 0.5)

        train_data = TensorDataset(x)
        train_loader = DataLoader(train_data, args.batch_size, shuffle=True, drop_last=True)
        test_loader = train_loader
        viz = None
        if "ising" in args.model:
            plot = lambda p, x: torchvision.utils.save_image(x.view(x.size(0), 1, args.dim_sqrt, args.dim_sqrt),
                                                             p, normalize=False, nrow=int(x.size(0) ** .5))
        elif args.model == "lattice_potts":
            plot = lambda p, x: torchvision.utils.save_image(
                x.view(x.size(0), args.dim_sqrt, args.dim_sqrt, 3).transpose(3, 1),
                p, normalize=False, nrow=int(x.size(0) ** .5))
        else:
            plot = lambda p, x: None
    else:
        raise ValueError

    return train_loader, test_loader, plot, viz


def generate_data(args):
    samples = None
    if args.data_model in ["ising_lattice", "ising_lattice_2d"]:
        model = LatticeIsingModel(args.dim_sqrt ** 2, args.sigma)
        sampler = BGAVSampler(n_dims=args.dim_sqrt ** 2, model_name=args.model)

    elif args.data_model == "ising_lattice_3d":
        model = LatticeIsingModel(args.dim_sqrt ** 2, args.sigma, lattice_dim=3)
        sampler = BGAVSampler(n_dims=args.dim_sqrt ** 2, model_name=args.model)
        print(model.sigma)
        print(model.G)
        print(model.J)

    elif args.data_model == "er_ising":
        model = ERIsingModel(args.dim_sqrt, args.degree, args.sigma)
        sampler = BGAVSampler(model_name=args.model)
        print(model.G)
        print(model.J)

    elif args.data_model == "ising_neural":
        assert args.model_load_path, "must provide model load path"
        model = QuadraticNeuralModel(args.dim_sqrt ** 2, learn_G=True, learn_bias=False, learn_net=False, net_weight=1.0)
        model.load_state_dict(torch.load(args.model_load_path))
        sampler = BinaryGWGSampler(model.data_dim)
        head, _ = os.path.split(args.model_load_path)
        with open(os.path.join(head, "buffer.npz")) as f:
            samples = np.load(f)["buffer"]
    else:
        raise ValueError

    model = model.to(args.device)
    if samples is None:
        samples = model.init_sample(args.n_samples).to(args.device)
    print("Generating {} samples from:".format(len(samples)))
    print(model)
    for _ in tqdm(range(args.gt_steps)):
        samples = sampler.step(samples, model).detach()

    return samples.detach().cpu(), model


def model_and_buffer(args, init_bias, data):

    ground_truth_J = None
    if args.model == "lattice_potts":
        model = LatticePottsModel(int(args.dim_sqrt), int(args.n_state), 0., 0., learn_sigma=True)

    elif args.model == "ising_lattice_2d":
        model = LatticeIsingModel(int(args.dim_sqrt ** 2), args.sigma, learn_G=True)
        ground_truth_J = model.J.clone().to(device)
        model.G.data = torch.randn_like(model.G.data) * .01
        model.sigma.data = torch.ones_like(model.sigma.data)

        plt.clf()
        plt.matshow(ground_truth_J.detach().cpu().numpy())
        plt.savefig("{}/ground_truth.png".format(args.save_dir))
        plt.savefig("{}/ground_truth.pdf".format(args.save_dir))

    elif args.model == "er_ising":
        model = ERIsingModel(int(args.dim_sqrt), 2, learn_G=True)
        model.G.data = torch.randn_like(model.G.data) * .01
        with open(args.graph_file, 'rb') as f:
            ground_truth_J = pickle.load(f)
            plt.clf()
            plt.matshow(ground_truth_J.detach().cpu().numpy())
            plt.savefig("{}/ground_truth.png".format(args.save_dir))
            plt.savefig("{}/ground_truth.pdf".format(args.save_dir))
        ground_truth_J = ground_truth_J.to(device)

    elif "neural" in args.model:
        if args.model_load_path:
            assert args.model == "ising_neural_frozen"
            model = QuadraticNeuralModel(args.dim_sqrt ** 2, learn_G=True, learn_bias=False, learn_net=False, net_weight=1.0)
            model.load_state_dict(torch.load(args.model_load_path))
            model.init_dist = torch.distributions.Bernoulli(probs=init_bias)
            ground_truth_J = model.J.to(device)
            plt.clf()
            plt.matshow(ground_truth_J.detach().cpu().numpy())
            plt.savefig("{}/ground_truth.png".format(args.save_dir))
            plt.savefig("{}/ground_truth.pdf".format(args.save_dir))

            model.to(device)
            # randomize ising params as we want to re-estimate these
            model.G.data = torch.randn_like(model.G.data) * .01
        else:
            # fit QuadEBM to real data (there is no ground truth J)
            model = QuadraticNeuralModel(args.dim_sqrt ** 2, learn_G=True, learn_bias=True,
                                         init_bias=init_bias, learn_net=True, net_weight=1.0)

    elif "boltzmann" in args.model:
        model = BoltzmannMachine(args.dim_sqrt**2, learn_G=True, learn_bias=True, init_bias=init_bias)

    buffer = model.init_sample(args.buffer_size)
    model.to(device)
    buffer = buffer.to(device)

    return model, buffer, ground_truth_J


def viz_diagnostics(args, get_J, ground_truth_J, itr, method_save_dir, model, rmses, sigmas, sq_errs):
    if args.model in ("lattice_potts", "ising_lattice"):
        sigmas.append(model.sigma.data.item())
        plt.clf()
        plt.plot(sigmas, label="model")
        plt.plot([args.sigma for s in sigmas], label="gt")
        plt.legend()
        plt.savefig("{}/sigma.png".format(method_save_dir))

    else:
        if ground_truth_J is not None:
            sq_err = ((ground_truth_J - get_J()) ** 2).sum()
            sq_errs.append(sq_err.item())
            plt.clf()
            plt.plot(sq_errs, label="sq_err")
            plt.legend()
            plt.savefig("{}/sq_err.png".format(method_save_dir))

            plt.clf()
            plt.plot(rmses, label="rmse")
            plt.legend()
            plt.savefig("{}/rmse.png".format(method_save_dir))

        plt.clf()
        plt.matshow(get_J().detach().cpu().numpy())
        plt.savefig("{}/model_{}.png".format(method_save_dir, itr))
        plt.clf()


def print_diagnostics(args, get_J, ground_truth_J, itr, itr_time, logp_fake, logp_real,
                      model, sampler, my_print, obj, rmses, rocaucs):
    my_print(f"({itr}) log p(real) = {logp_real.item():.4f},"
             f" log p(fake) = {logp_fake.item():.4f}, "
             f"diff = {obj.item():.4f},"
             f" prop_hops = {sampler.proposed_hops[-1]:.4f},"
             f"acc rate = {sampler.acc_rates[-1]:.4f},"
             f"itr time = {itr_time:.4f}"
             )

    if args.model in ("lattice_potts", "ising_lattice"):
        my_print("\tsigma true = {:.4f}, current sigma = {:.4f}".format(args.sigma, model.sigma.data.item()))

    if ground_truth_J is not None:
        rocauc = graph_rocauc(numpify(ground_truth_J), numpify(get_J()))
        rocaucs.append(rocauc)
        # diagonal of Ising matrix has no effect on unnormalised log prob so ignore it
        I = torch.eye(len(ground_truth_J), device=ground_truth_J.device)
        rmse = (((ground_truth_J - get_J()) ** 2) * (1 - I)).mean().sqrt()
        rmses.append(rmse.item())
        my_print("\t rocauc = {:.4f}, rmse = {:.4f}".format(rocauc, rmse))


def save_everything(args, buffer, get_J, ground_truth_J, method_save_dir, model, rmses,
                    rocaucs, log_mmds, tstamp, av_time_per_sample_step, total_time, itr):

    with open("{}/time_per_sample_iter.txt".format(method_save_dir), 'w') as f:
        f.write(str(av_time_per_sample_step))

    with open(f"{method_save_dir}/buffer.npz", 'wb') as f:
        np.savez(f, buffer=numpify(buffer))

    if "neural" in args.model:
        torch.save(model.state_dict(), os.path.join(method_save_dir, f"{args.model}_model_{tstamp}"))
    else:
        np.savez(os.path.join(method_save_dir, "J"), J=numpify(model.J))

    if args.model == "lattice_potts":
        final_sigma = model.sigma.data.item()
        with open("{}/sigma.txt".format(method_save_dir), 'w') as f:
            f.write(str(final_sigma))
    else:
        try:
            with open(os.path.join(method_save_dir, "rmses_per_iter.npz"), 'wb') as f:
                np.savez(f, rmses=np.array(rmses), total_time=total_time, total_itrs=itr)

            with open(os.path.join(method_save_dir, "rocaucs_per_iter.npz"), 'wb') as f:
                np.savez(f, rocaucs=np.array(rocaucs), total_time=total_time, total_itrs=itr)

            with open(os.path.join(method_save_dir, "log_mmds_per_iter.npz"), 'wb') as f:
                np.savez(f, log_mmds=np.array(log_mmds), total_time=total_time, total_itrs=itr)

            sq_err = ((ground_truth_J - get_J()) ** 2).sum().item()
            rmse = ((ground_truth_J - get_J()) ** 2).mean().sqrt().item()
            with open("{}/sq_err.txt".format(method_save_dir), 'w') as f:
                f.write(str(sq_err))
            with open("{}/rmse.txt".format(method_save_dir), 'w') as f:
                f.write(str(rmse))

        except Exception as e:
            print(e)
            pass

    print(f"Time passed: {total_time}")


def setup_data_dist(args, my_print):

    if args.data_file is not None:
        # load existing data
        my_print(f"Loading data from {args.data_file}")
        train_loader, test_loader, plot, viz = get_data(args)
        return train_loader, test_loader, plot

    else:
        assert args.data_save_dir is not None, \
            "About to generate data, but do not have a path to save it to since args.data_save_dir is None. " \
            "Please either specify this, or specify a path for loading data by setting args.data_file."

        # generate data & quit
        data, data_model = generate_data(args)

        my_print(f"We have created your data. Saving to {args.data_save_dir}/data.pkl")
        with open("{}/data.pkl".format(args.data_save_dir), 'wb') as f:
            pickle.dump(data, f)

        if args.data_model == "er_ising":
            ground_truth_J = data_model.J.detach().cpu()
            with open("{}/J.pkl".format(args.data_save_dir), 'wb') as f:
                pickle.dump(ground_truth_J, f)

        my_print(
            f"Exiting this script. If you would like to train a model on the data just created, re-run this script "
            f"and set args.data_file to {args.data_save_dir}/data.pkl")
        quit()


def main(args):
    makedirs(args.save_dir)
    if args.data_save_dir: makedirs(args.data_save_dir)
    method_save_dir = os.path.join(args.save_dir, args.sampler, str(args.sampling_steps_per_iter))
    if args.model_load_path: method_save_dir = os.path.join(method_save_dir, "reestimated")
    os.makedirs(method_save_dir, exist_ok=True)
    logger = open("{}/log.txt".format(args.save_dir), 'w')
    tstamp = strftime("%Y-%m-%d_%H-%M-%S", gmtime())

    def my_print(s):
        print(s)
        logger.write(str(s) + '\n')

    # make G symmetric
    def get_J():
        j = model.J
        return (j + j.t()) / 2

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_loader, test_loader, plot = setup_data_dist(args, my_print)

    trn_data = get_all_data(train_loader)
    trn_500 = trn_data[:500]
    tst_500 = get_all_data(test_loader)[:500]
    kmmd = mmd.MMD(mmd.exp_avg_hamming, use_ustat=False)
    opt_mmd = kmmd.compute_mmd(trn_500, tst_500)
    opt_log_mmd = opt_mmd.log10().item()
    print("gt <--> gt log-mmd", opt_mmd, opt_log_mmd)
    precon_mat = torch.cov(trn_data.T)
    if args.sampler.lower() == "pavg" and args.precon_load_path:
        print(f"Loading precon matrix from {args.precon_load_path}")
        precon_mat = torchify(np.load(args.precon_load_path)["J"])

    ###### MODEL & BUFFER ######
    args.data_dim = args.dim_sqrt ** 2
    model, buffer, ground_truth_J = model_and_buffer(args, init_bias=trn_data.mean(0), data=trn_data[:100])

    ###### SAMPLER ######
    method_dict = {'name': args.sampler,
                   'epsilon': args.epsilon,
                   'allow_adaptation_of_precon_matrix': False,
                   'init_precon_mat': precon_mat,
                   'n_forward_copies': 5,
                   }

    sampler = get_sampler(args, method_dict, method_save_dir)

    ###### OPTIMIZER ######
    all_params = set(model.parameters())
    wd_params = set(model.net.parameters()) if args.model == "ising_neural" else set()
    no_wd = all_params - wd_params
    optimizer = torch.optim.AdamW(
        [
            {'params': list(no_wd), 'weight_decay': 0},
            {'params': list(wd_params)}
        ],
        lr=args.lr, eps=1e-7, weight_decay=args.weight_decay
    )

    my_print(device)
    my_print(model)
    my_print(buffer.size())
    my_print(sampler)

    itr = 0
    sigmas = []
    sq_errs = []
    rmses = []
    rocaucs = []
    log_mmds = []
    buffer_idx = 0
    total_time = 0.0
    while itr <= args.n_iters:
        for x in train_loader:
            itr_start_time = time()
            x = x[0].to(device)

            ###### UPDATE PCD BUFFER ######
            buffer_idx += args.batch_size
            if buffer_idx >= len(buffer):
                buffer_idx = 0
                buffer = buffer[torch.randperm(len(buffer))]
            if hasattr(sampler, "cache"):
                sampler.cache = None  # model has changed, so delete cache
            for k in range(args.sampling_steps_per_iter):
                stop = min(buffer_idx + args.batch_size, len(buffer))
                buffer[buffer_idx:stop] = sampler.step(buffer.detach()[buffer_idx:stop], model).detach()

            ###### UPDATE MODEL ######
            logp_real = model(x).squeeze().mean()
            logp_fake = model(buffer[buffer_idx:stop]).squeeze().mean()
            obj = logp_real - logp_fake
            loss = -obj
            loss += args.l1 * get_J().abs().sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_time += time() - itr_start_time

            ######### DIAGNOSTICS ##############
            if itr % args.print_every == 0:
                print_diagnostics(args, get_J, ground_truth_J, itr, total_time / (itr + 1), logp_fake, logp_real,
                                  model, sampler, my_print, obj, rmses, rocaucs)

            if itr % args.viz_every == 0:
                diagnose_mmd(buffer, kmmd, method_save_dir, log_mmds, my_print, opt_log_mmd, trn_500)
                viz_diagnostics(args, get_J, ground_truth_J, itr, method_save_dir, model, rmses, sigmas, sq_errs)
                plot("{}/data_{}.png".format(method_save_dir, itr), x.detach().cpu())
                plot("{}/buffer_{}.png".format(method_save_dir, itr), buffer[:args.batch_size].detach().cpu())
                plt.clf()

            if itr > 0 and itr % 1000 == 0:
                save_everything(args, buffer, get_J, ground_truth_J, method_save_dir, model, rmses, rocaucs, log_mmds,
                                tstamp, sampler.av_time, total_time, itr)

            itr += 1

    save_everything(args, buffer, get_J, ground_truth_J, method_save_dir,
                    model, rmses, rocaucs, log_mmds, tstamp, sampler.av_time, total_time, itr)


def parse_args():
    # model = "ising_lattice_2d"
    # model = "boltzmann"
    # model = "ising_neural"
    model = "ising_neural_frozen"
    l1_reg = 0.01
    n_iters = 5100 if model in ["ising_neural"] else 2100
    sigma, data_save_dir, model_load_path = None, None, None
    if model == "ising_lattice_2d":
        """Data distribtuion and learned model are lattice ising models. 
        To generate ground-truth data, run script with data_file=None and data_save_dir="YOUR_DATA_PATH".
        To then fit a model to this ground truth data, re-run the script, but this time with data_file="YOUR_DATA_PATH/data.pkl"
        """
        sigma = 0.2
        data_file = os.path.join(os.getcwd(), "data", f"ising_lattice_sigma{sigma}", "data.pkl")
        savedir = os.path.join(os.getcwd(), "results", f"ising_lattice_sigma{sigma}")

    elif model in ["ising_neural_frozen", "ising_neural"]:
        """If data_file is None, then we will call generate_data(), save that data to data_save_dir and then quit"""
        """if model_load_path != None, then we load a model & attempt to re-estimate (all or part of) the model"""
        dname = 'usps'  # usps, histopathology, bsds300
        # data_file = dname
        data_file = os.path.join(os.getcwd(), "results", f"ising_neural_{dname}", "gwg", "50", "buffer.npz")
        model_load_path = os.path.join(os.getcwd(), "results", f"ising_neural_{dname}", "gwg", "50",
                                       "ising_neural_model_2022-07-10_10-41-16")
        savedir = os.path.join(os.getcwd(), "results", f"{model}_{dname}")
    elif model in ["boltzmann"]:
        dname = 'usps'  # usps, histopathology, bsds300
        # data_file = dname
        data_file = os.path.join(os.getcwd(), "results", f"ising_neural_{dname}", "gwg", "50", "buffer.npz")
        savedir = os.path.join(os.getcwd(), "results", f"{model}_{dname}")
        l1_reg = 0.0
    else:
        raise NotImplementedError

    parser = argparse.ArgumentParser()

    ##### I/O ######
    parser.add_argument('--save_dir', type=str, default=savedir)
    parser.add_argument('--data_file', type=str, default=data_file,
                        help="path to data file. If none, then this script will call generate_data() and then quit.")
    parser.add_argument('--data_save_dir', type=str, default=data_save_dir,
                        help="only used if data_file is None, since then we call generate_data() and save it to this directory")
    parser.add_argument('--graph_file', type=str, help="location of pkl containing graph")

    ##### data generation #####
    parser.add_argument('--var_type', type=str, default="binary")
    parser.add_argument('--gt_steps', type=int, default=50000)
    parser.add_argument('--n_samples', type=int, default=10000)
    parser.add_argument('--sigma', type=float, default=sigma)  # ising and potts
    parser.add_argument('--degree', type=int, default=2)  # ER model
    parser.add_argument('--data_model', type=str, default=model,
                        choices=['ising_lattice_2d', 'lattice_potts', 'er_ising', 'ising_neural'],
                        help="only used when generating data i.e. data_save_dir is not None."
                        )

    ##### Model args ######
    parser.add_argument('--model',
                        choices=['ising_lattice_2d', 'lattice_potts', 'er_ising', 'ising_neural', 'ising_neural_frozen'],
                        type=str, default=model)
    parser.add_argument('--model_load_path', type=str, default=model_load_path)
    parser.add_argument('--precon_load_path', type=str, default=None)
    # parser.add_argument('--precon_load_path', type=str,
    #                     default=os.path.join(os.getcwd(), "results", f"boltzmann_{dname}", "block-gibbs", "1", "J.npz"))

    ##### Sampler args ######
    parser.add_argument('--sampler', type=str, default='NCG')
    # parser.add_argument('--sampler', type=str, default='GWG')
    # parser.add_argument('--sampler', type=str, default='PAVG')
    # parser.add_argument('--sampler', type=str, default='block-gibbs')
    # parser.add_argument('--sampling_steps_per_iter', type=int, default=1)
    parser.add_argument('--sampling_steps_per_iter', type=int, default=5)
    parser.add_argument('--epsilon', type=float, default=0.2, help="initial epsilon value")

    ##### PCD learning args ######
    parser.add_argument('--n_iters', type=int, default=n_iters)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--buffer_size', type=int, default=5000)
    parser.add_argument('--weight_decay', type=float, default=1e-4 if model == "ising_neural" else 0.0,
                        help="regularisation for Neural params (if using)")
    parser.add_argument('--l1', type=float, default=l1_reg, help="regularisation for Ising matrix")

    ###### Misc args ######
    parser.add_argument('--seed', type=int, default=123456)
    parser.add_argument('--approx', action="store_true")
    parser.add_argument('--n_hidden', type=int, default=25)
    parser.add_argument('--dim_sqrt', type=int, default=10)
    parser.add_argument('--n_state', type=int, default=3)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--viz_batch_size', type=int, default=1000)
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--viz_every', type=int, default=100)
    args = parser.parse_args()
    args.device = device

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
