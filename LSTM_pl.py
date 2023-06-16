import utils
import torch
import torch.nn as nn
import random
import torch.optim as optim
import tqdm
import datetime
import tqdm
import matplotlib.pyplot as plt
import numpy as np

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm_cell = nn.LSTMCell(input_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_seq, predict_len,lanes,lane_norms,scene_num):
        device = input_seq.device
        
        first_col = input_seq[:, 0, :2].clone()
        broadcasted_first_col = first_col.unsqueeze(1).expand(-1, input_seq.shape[1], -1)
        input_seq[:, :, :2] -=  broadcasted_first_col

        (batch_size,seq_len,fea_len) = input_seq.shape
        input_seq = input_seq.permute(1,0,2)
        h,c = torch.zeros(batch_size, self.hidden_dim, device=device),torch.zeros(batch_size, self.hidden_dim,device=device)
        outputs = []

        for i in range(seq_len):
            input = input_seq[i]
            # for j in range(batch_size):
            # print(lanes[0].shape)
            idx = 0
            nearest_lanes,nearest_lane_norms = [],[]
            for j in range(len(scene_num)):
                nearest_lane,nearest_lane_norm = utils.get_nearest_lane(input[idx:idx+scene_num[j]],lanes[j],lane_norms[j])
                nearest_lanes.append(nearest_lane)
                nearest_lane_norms.append(nearest_lane_norm)
                idx += scene_num[j]
            nearest_lanes = torch.cat(nearest_lanes,dim=0)
            nearest_lane_norms = torch.cat(nearest_lane_norms,dim=0)
            # nearest_lane_norm = nearest_lane_norm/(torch.norm(nearest_lane_norm, dim=1).unsqueeze(1))
            # input = torch.cat([input,nearest_lane_norms],dim=1)
            l = nearest_lanes - nearest_lane_norms/2 -first_col
            r = nearest_lanes + nearest_lane_norms/2 -first_col
            # print(l,r)
            input = torch.cat([input,l,r],dim=1)
            h, c = self.lstm_cell(input, (h, c))

        for i in range(predict_len):
            output = self.linear(h)
            outputs.append(output)

            idx = 0
            nearest_lanes,nearest_lane_norms = [],[]
            for j in range(len(scene_num)):
                nearest_lane,nearest_lane_norm = utils.get_nearest_lane(output[idx:idx+scene_num[j]],lanes[j],lane_norms[j])
                nearest_lanes.append(nearest_lane)
                nearest_lane_norms.append(nearest_lane_norm)
                idx += scene_num[j]
            nearest_lanes = torch.cat(nearest_lanes,dim=0)
            nearest_lane_norms = torch.cat(nearest_lane_norms,dim=0)
            # nearest_lane_norm = nearest_lane_norm/(torch.norm(nearest_lane_norm, dim=1).unsqueeze(1))
            # input = torch.cat([input,nearest_lane_norms],dim=1)
            l = nearest_lanes - nearest_lane_norms/2 -first_col
            r = nearest_lanes + nearest_lane_norms/2 -first_col
            # print(l,r)
            output = torch.cat([output,l,r],dim=1)

            h, c = self.lstm_cell(output, (h, c))

        outputs = torch.stack(outputs, dim=0)
        outputs = outputs.permute(1,0,2)

        broadcasted_first_col = first_col.unsqueeze(1).expand(-1, outputs.shape[1], -1)
        outputs += broadcasted_first_col

        return outputs
    
def lane2p(lanes,lane_norms):
    new_lanes = []
    for i in range(len(lanes)):
        l = torch.tensor(lanes[i] - lane_norms[i]/2)
        r = torch.tensor(lanes[i] + lane_norms[i]/2)
        new_lane = torch.cat([l,r],dim=-1)
        new_lanes.append(new_lane)
        return new_lanes

data_path = "C:\\Users\\zxk\\Desktop\\251B\\class-proj\\ucsd-cse-251b-class-competition\\"
city_idx_path = "C:\\Users\\zxk\\Desktop\\251B\\class-proj\\cse251b-project\\"
model_path = "C:\\Users\\zxk\\Desktop\\251B\\class-proj\\model\\LSTM_PL\\"
mode = "train"
batch_size = 64
cutoff = None
collate_fn = utils.collate_with_lane
MIA_train_loader,PIT_train_loader,MIA_valid_loader,PIT_valid_loader,MIA_train_dataset,PIT_train_dataset,MIA_valid_dataset,PIT_valid_dataset = utils.loadData(data_path,city_idx_path,batch_size,split=0.9,cutoff=cutoff,collate_fn=collate_fn)

input_size = 6
hidden_size = 200
output_size = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

'''all agents as a sample'''
# input_size = 60 * 4
# hidden_size = 200
# output_size = 60 * 4

# learning_rate = 1E-2
# epochs = 10

# model = LSTM(input_dim=input_size,hidden_dim=hidden_size,output_dim=output_size)

# optimizer = optim.Adam(model.parameters(),lr = learning_rate)
# criterion = nn.MSELoss()

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Using device:', device)
# model = model.to(device)
# losses = []

# for epoch in range(epochs):
#     eloss = []
#     for i_batch, sample_batch in enumerate(MIA_train_loader):
#         inp, out = sample_batch # [batch_size, track_sum, seq_len, features]
#         inp = inp.permute(0,2,1,3)
#         out = out.permute(0,2,1,3)
#         inp = inp.reshape(inp.shape[0],inp.shape[1],-1).float()
#         out = out.reshape(out.shape[0],out.shape[1],-1).float()
#         inp,out = inp.to(device),out.to(device)
#         predict_len = out.shape[1]

#         optimizer.zero_grad()
#         predict = model(inp,predict_len)
#         loss = torch.sqrt(criterion(out,predict))
#         loss.backward()
#         optimizer.step()
#         eloss.append(loss.item())
#     avgloss = sum(eloss)/len(eloss)
#     print("Epoch:",epoch,"Loss:",loss)
#     losses.append(avgloss)

'''an agent as a example'''

if mode == "train":
    learning_rate = 2E-4
    epochs = 6

    model = LSTM(input_dim=input_size,hidden_dim=hidden_size,output_dim=output_size)
    print("Parameter:",sum(p.numel() for p in model.parameters() if p.requires_grad))
    model.load_state_dict(torch.load(model_path+'2023-06-03_03-06-25_model_6.pth'))

    optimizer = optim.Adam(model.parameters(),lr = learning_rate)
    criterion = nn.MSELoss()

    model = model.to(device)
    model.train()
    losses = []

    progress_bar = tqdm.tqdm(range(epochs))

    print("---start train---")

    for epoch in progress_bar:
        eloss = []
        for i_batch, sample_batch in enumerate(MIA_train_loader):
            inp, out,mask,lanes,lane_norms = sample_batch # [batch_size, track_sum, seq_len, features]
            # lane_segments = lane2p(lane,lane_norm)
            scene_num = torch.sum(mask,dim=1).int()
            mask = mask.ravel()
            indices = torch.nonzero(mask).squeeze()
            inp, out = inp.reshape(-1,inp.shape[2],inp.shape[3]).float(),out.reshape(-1,out.shape[2],out.shape[3]).float()
            inp, out = inp[indices],out[indices]
            inp, out = inp.to(device),out.to(device)
            lanes = [it.to(device) for it in lanes]
            lane_norms  = [it.to(device) for it in lane_norms]
            # print(inp.shape,out.shape)
            predict_len = out.shape[1]
            # first_col = inp[:, 0, :2].clone()
            # broadcasted_first_col = first_col.unsqueeze(1).expand(-1, inp.shape[1], -1)
            # inp[:, :, :2] -=  broadcasted_first_col

            inp, out = inp[:, :, :2], out[:, :, :2]
            optimizer.zero_grad()

            predict = model(inp,predict_len,lanes,lane_norms,scene_num)
            # broadcasted_first_col = first_col.unsqueeze(1).expand(-1, predict.shape[1], -1)
            # predict += broadcasted_first_col
            # print(predict.shape,out.shape)
            loss = criterion(out,predict)
            loss.backward()
            optimizer.step()
            eloss.append(loss.item())
            if (i_batch+1) % 100 == 0:
                 progress_bar.set_description("Epoch: {} Batch: {} Loss {:.4f}".format(epoch,i_batch+1,loss))  
            # break
        avgloss = sum(eloss)/len(eloss)
        # progress_bar.set_description("Epoch {} Train loss: {:.4f}".format(epoch+1,avgloss))
        # print("Epoch:",epoch+1,"Loss:",loss)
        losses.append(avgloss)
        
        current_datetime = datetime.datetime.now()
        current_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

        if (epoch + 1) % 1 == 0:
            torch.save(model.state_dict(), model_path+str(current_datetime)+'_model_'+str(epoch+1)+'.pth')
        # break
    # print(predict,out)
    plt.plot(losses)
    plt.show()

if mode == "test":
    model = LSTM(input_dim=input_size,hidden_dim=hidden_size,output_dim=output_size)
    model.load_state_dict(torch.load(model_path+'2023-06-03_14-29-29_model_6.pth'))

    model = model.to(device)

    model.eval()

    criterion = nn.MSELoss()

    tlosses = []

    for i_batch, sample_batch in enumerate(MIA_valid_loader):
        inp, out,mask,lanes,lane_norms = sample_batch  # [batch_size, track_sum, seq_len, features]
        scene_num = torch.sum(mask,dim=1).int()
        mask = mask.ravel()
        indices = torch.nonzero(mask).squeeze()
        inp, out = inp.reshape(-1,inp.shape[2],inp.shape[3]).float(),out.reshape(-1,out.shape[2],out.shape[3]).float()
        inp, out = inp[indices],out[indices]
        lanes = [it.to(device) for it in lanes]
        lane_norms  = [it.to(device) for it in lane_norms]
        # print(sum(mask),inp.shape[0])
        inp,out = inp.to(device),out.to(device)

        predict_len = out.shape[1]

        inp, out = inp[:, :, :2], out[:, :, :2]

        predict = model(inp,predict_len,lanes,lane_norms,scene_num)

        out = out[:,:,:2]
        predict = predict[:,:,:2]
        loss = criterion(out,predict)
        tlosses.append(loss.item())

        # break
    # print(predict,out)

    print("Average MSE Loss: ",sum(tlosses)/len(tlosses))

if mode == "visual":

    model = LSTM(input_dim=input_size,hidden_dim=hidden_size,output_dim=output_size)
    model.load_state_dict(torch.load(model_path+'2023-06-02_15-32-55_model_6.pth'))
    model = model.to("cpu")
    print(model)

    sample_idx = 99
    traj_idx = 1

    sample = MIA_valid_dataset[sample_idx]

    inp = np.dstack([sample["p_in"], sample["v_in"]])
    
    lanes = [sample["lane"]]

    lane_norms = [sample["lane_norm"]]

    lanes,lane_norms = torch.tensor(lanes).float(),torch.tensor(lane_norms).float()

    scene_num = [1]

    # mask = torch.tensor(sample["car_mask"]).ravel()

    # indices = torch.nonzero(mask).squeeze()

    inp = torch.tensor(inp[traj_idx:traj_idx+1]).float()

    predict_len = 30

    inp = inp[:, :, :2]

    predict = model(inp,predict_len,lanes,lane_norms,scene_num)

    pred_X = predict[0,:,0]
    pred_Y = predict[0,:,1]

    utils.visualization(sample,pred_X.detach(),pred_Y.detach(),traj_idx)
    
if mode == "output":

    model = LSTM(input_dim=input_size,hidden_dim=hidden_size,output_dim=output_size)
    model.load_state_dict(torch.load(model_path+'2023-06-03_14-29-29_model_6.pth'))

    model = model.to(device)

    path = 'C:\\Users\\zxk\\Desktop\\251B\\class-proj\\ucsd-cse-251b-class-competition\\val_in\\val_in'

    scence_ids,inp,lanes,lane_norms = utils.loadValidData_by_traj_lane(path)
    scene_num = [1 for i in range(len(lanes))]
    
    lanes = [it.to(device) for it in lanes]
    lane_norms  = [it.to(device) for it in lane_norms]
    inp = inp.float().to(device)

    predict_len = 30

    inp = inp[:,:,:2]
    predict = model(inp,predict_len,lanes,lane_norms,scene_num)

    path = "C:\\Users\\zxk\\Desktop\\251B\\class-proj\\ucsd-cse-251b-class-competition\\"
    name = "LSTM_MIA_PL_3.csv"

    utils.formOutput(path,predict[:,:,:2],scence_ids,name)

    