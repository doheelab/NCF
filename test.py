import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

import model
import config
import evaluate
import data_utils


args = {
    "batch_size": 256,
    "dropout": 0.0,
    "epochs": 20,
    "factor_num": 32,
    "gpu": "0",
    "lr": 0.001,
    "num_layers": 3,
    "num_ng": 4,
    "out": True,
    "test_num_ng": 99,
    "top_k": 10,
}
os.environ["CUDA_VISIBLE_DEVICES"] = args["gpu"]
cudnn.benchmark = True


############################## PREPARE DATASET ##########################
train_data, test_data, user_num, item_num, train_mat = data_utils.load_all()

# train_data = [[0, 121], [0, 199], [1, 456],...]


# construct the train and test datasets
train_dataset = data_utils.NCFData(
    train_data, item_num, train_mat, args["num_ng"], True
)
test_dataset = data_utils.NCFData(test_data, item_num, train_mat, 0, False)
train_loader = data.DataLoader(
    train_dataset, batch_size=args["batch_size"], shuffle=True, num_workers=4
)
test_loader = data.DataLoader(
    test_dataset, batch_size=args["test_num_ng"] + 1, shuffle=False, num_workers=0
)

########################### CREATE MODEL #################################
if config.model == "NeuMF-pre":
    assert os.path.exists(config.GMF_model_path), "lack of GMF model"
    assert os.path.exists(config.MLP_model_path), "lack of MLP model"
    GMF_model = torch.load(config.GMF_model_path)
    MLP_model = torch.load(config.MLP_model_path)
else:
    GMF_model = None
    MLP_model = None

model = model.NCF(
    user_num,
    item_num,
    args["factor_num"],
    args["num_layers"],
    args["dropout"],
    config.model,
    GMF_model,
    MLP_model,
)
model.cuda()
loss_function = nn.BCEWithLogitsLoss()

if config.model == "NeuMF-pre":
    optimizer = optim.SGD(model.parameters(), lr=args["lr"])
else:
    optimizer = optim.Adam(model.parameters(), lr=args["lr"])

# writer = SummaryWriter() # for visualization

########################### TRAINING #####################################

if __name__ == "__main__":
    count, best_hr = 0, 0
    for epoch in range(args["epochs"]):
        model.train()  # Enable dropout (if have).
        start_time = time.time()
        train_loader.dataset.ng_sample()

        for user, item, label in train_loader:
            user = user.cuda()
            item = item.cuda()
            label = label.float().cuda()

            model.zero_grad()
            prediction = model(user, item)
            loss = loss_function(prediction, label)
            loss.backward()
            optimizer.step()
            # writer.add_scalar('data/loss', loss.item(), count)
            count += 1

        model.eval()
        HR, NDCG = evaluate.metrics(model, test_loader, args["top_k"])

        elapsed_time = time.time() - start_time
        print(
            "The time elapse of epoch {:03d}".format(epoch)
            + " is: "
            + time.strftime("%H: %M: %S", time.gmtime(elapsed_time))
        )
        print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

        if HR > best_hr:
            best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
            if args["out"]:
                if not os.path.exists(config.model_path):
                    os.mkdir(config.model_path)
                torch.save(model, "{}{}.pth".format(config.model_path, config.model))

    print(
        "End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(
            best_epoch, best_hr, best_ndcg
        )
    )
