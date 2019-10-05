import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchvision import models

class VADPredict(nn.Module):
    """
    VADを予測するネットワーク
    """
    def __init__(self):
        super(VADPredict, self).__init__()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 32)
        self.fc4 = nn.Linear(32, 2)

    def forward(self,x):
        h = F.dropout(F.relu(self.fc1(x)),p=0.5)
        h = F.dropout(F.relu(self.fc2(x)), p=0.5)
        h = F.dropout(F.relu(self.fc3(x)), p=0.5)
        h = self.fc4(h)
        return h

    def get_middle(self,x):
        h = F.dropout(F.relu(self.fc1(x)), p=0.5)
        h = F.dropout(F.relu(self.fc2(x)), p=0.5)
        h = F.dropout(F.relu(self.fc3(x)), p=0.5)
        return h

class TimeVADPredict(nn.Module):
    """
    VAD予測するネットワーク
    """
    def __init__(self, num_layers = 1, hidden_size = 256):
        super(TimeVADPredict, self).__init__()
        self.lstm = torch.nn.LSTM(
            input_size = hidden_size, #入力size
            hidden_size = hidden_size, #出力size
            num_layers = 1, #stackする数
            dropout = 0.5,
            batch_first = True, # given_data.shape = (batch , frames , input_size)
            bidirectional = False
        )
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 2)

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x, h, c):
        assert len(x.shape) == 3 , print('data shape is incorrect.')
        batch_size, frames = x.shape[:2]
        if h is None:
            h,c = self.reset_state(batch_size)
            print('reset state!!')
        #self.reset_state(batch_size)
        h, (h0, c0) = self.lstm(x, (h,c))
        #h, hidden = self.lstm(x, h)

        h = F.dropout(F.relu(self.fc1(h[:,-1,:])),p=0.5)
        y = self.fc2(h)

        return y, h0, c0



    def reset_state(self, batch_size):
        self.h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        self.c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        return self.h0,self.c0
    def set_state(self,h,c):
        self.h0 = h
        self.c0 = c

    def get_middle(self,x):
        batch_size, frames = x.shape[:2]
        if h is None:
            h, c = self.reset_state(batch_size)
            print('reset state!!')
        self.reset_state(batch_size)
        h, (h0, c0) = self.lstm(x, (h, c))
        # h, hidden = self.lstm(x, h)

        h = F.dropout(F.relu(self.fc1(h[:, -1, :])), p=0.5)

        return h, h0, c0

class GazeTrain(nn.Module):
    """
    Resnet150の最終層をLSTMに入力して状態を推定するモデル
    """

    def __init__(self, num_layers=1, hidden_size=256):
        super(GazeTrain, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=32,
                               kernel_size=5,
                               stride=1,
                               padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32,32,5,1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(32,32,3,1)
        self.conv4 = nn.Conv2d(32,32,3,1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32,32,3,1)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(6*38*32, 1024)
        self.fc2 = nn.Linear(1024, 64)
        self.fc3 = nn.Linear(64, 2)

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        for f in [self.conv1, self.relu, self.conv2, self.relu, self.bn1, self.pool1, self.conv3, self.relu, self.conv4, self.relu, self.bn2, self.conv5, self.relu, self.bn3]:
            x = f(x)

        x = x.view(len(x),-1)
        x = F.dropout(F.relu(self.fc1(x)),p=0.5)
        h = F.dropout(F.relu(self.fc2(x)),p=0.5)
        y = self.fc3(h)
        return y

    def get_middle(self,x):
        for f in [self.conv1, self.relu, self.conv2, self.relu, self.bn1, self.pool1, self.conv3, self.relu, self.conv4, self.relu, self.bn2, self.conv5, self.relu, self.bn3]:
            x = f(x)

        x = x.view(len(x),-1)
        x = F.dropout(F.relu(self.fc1(x)),p=0.5)
        h = F.dropout(F.relu(self.fc2(x)),p=0.5)
        return h


class TargetPredict(nn.Module):
    """
    Targetを予測するネットワーク
    """
    def __init__(self,num_layers=1,hidden_size=128):
        super(TargetPredict, self).__init__()
        self.lstm = torch.nn.LSTM(
            input_size=129,  # 入力size
            hidden_size=hidden_size,  # 出力size
            num_layers=num_layers,  # stackする数
            dropout=0.5,
            batch_first=True,  # given_data.shape = (batch , frames , input_size)
            bidirectional=False
        )
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)
        self.vad = VADPredict()
        self.gaze = GazeTrain()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x_img, x_spA, x_spB, x_t, h, c):
        h_A = self.vad.get_middle(x_spA)
        h_B = self.vad.get_middle(x_spB)
        h_i = self.gaze.get_middle(x_img)
        #print(h_A.shape,h_i.shape,x_t.shape)
        #TO DO : h_A.h_B,h_i をconcat
        x = torch.cat((h_A, h_B, h_i, x_t.unsqueeze(0)), dim=1)
        x = x.view(len(x), 1, -1)

        batch_size, frames = x.shape[:2]
        if h is None:
            h, c = self.reset_state(batch_size)
            print('reset state!!')
        h, (h0, c0) = self.lstm(x, (h, c))

        h = F.dropout(F.relu(self.fc1(h[:, -1, :])), p=0.5)
        y = self.fc2(h)

        return y, h0, c0

    def reset_state(self, batch_size):
        self.h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        self.c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        return self.h0,self.c0

class TargetPredictLLD(nn.Module):
    """
    Targetを予測するネットワーク
    """
    def __init__(self,num_layers=1,hidden_size=128):
        super(TargetPredictLLD, self).__init__()
        self.lstm = torch.nn.LSTM(
            input_size=129,  # 入力size
            hidden_size=hidden_size,  # 出力size
            num_layers=num_layers,  # stackする数
            dropout=0.5,
            batch_first=True,  # given_data.shape = (batch , frames , input_size)
            bidirectional=False
        )
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)
        self.vad = VADPredictLLD()
        self.gaze = GazeTrain()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x_img, x_spA, x_spB, x_t, h, c):
        h_A = self.vad.get_middle(x_spA)
        h_B = self.vad.get_middle(x_spB)
        h_i = self.gaze.get_middle(x_img)

        x = torch.cat((h_A, h_B, h_i, x_t.unsqueeze(0)), dim=1)
        x = x.view(len(x), 1, -1)

        batch_size, frames = x.shape[:2]
        if h is None:
            h, c = self.reset_state(batch_size)
            print('reset state!!')
        h, (h0, c0) = self.lstm(x, (h, c))

        h = F.dropout(F.relu(self.fc1(h[:, -1, :])), p=0.5)
        y = self.fc2(h)

        return y, h0, c0

    def reset_state(self, batch_size):
        self.h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        self.c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        return self.h0,self.c0

class ResponsePredict(nn.Module):
    """
    Targetを予測するネットワーク
    """
    def __init__(self,num_layers=1,hidden_size=64):
        super(ResponsePredict, self).__init__()
        self.lstm = torch.nn.LSTM(
            input_size=66,  # 入力size
            hidden_size=hidden_size,  # 出力size
            num_layers=num_layers,  # stackする数
            dropout=0.5,
            batch_first=True,  # given_data.shape = (batch , frames , input_size)
            bidirectional=False
        )
        self.fc1 = nn.Linear(128, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)
        self.vad = VADPredict()
        self.vad.load_state_dict(torch.load('./vad_model.pth',map_location=torch.device('cpu')))
        self.gaze = GazeTrain()
        self.gaze.load_state_dict(torch.load('./gaze_model.pth',map_location=torch.device('cpu')))

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
    def forward(self, x_img, x_spA, x_spB, x_t, h, c, y_past):
        h_A = self.vad.get_middle(x_spA)
        h_B = self.vad.get_middle(x_spB)
        h_i = self.gaze.get_middle(x_img)
        #print(h_A.shape,h_i.shape,x_t.shape)
        #TO DO : h_A.h_B,h_i をconcat
        x = torch.cat((h_A, h_B, h_i),dim=1)
        x = F.dropout(F.relu(self.fc1(x)), p=0.5)
        x = torch.cat((x, x_t.unsqueeze(0),y_past.unsqueeze(0)), dim=1)
        #x = torch.cat((h_A, h_B, h_i, x_t.unsqueeze(0),y_past.unsqueeze(0)), dim=1)
        x = x.view(len(x), 1, -1)

        batch_size, frames = x.shape[:2]
        if h is None:
            h, c = self.reset_state(batch_size)
            print('reset state!!')
        # self.reset_state(batch_size)
        h, (h0, c0) = self.lstm(x, (h, c))

        #h = F.dropout(F.relu(self.fc1(h[:, -1, :])), p=0.5)
        y = self.fc2(h[:, -1, :])

        return y, h0, c0

    def reset_state(self, batch_size):
        self.h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        self.c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        return self.h0,self.c0


class MovePredict(nn.Module):
    """
    VAD予測するネットワーク
    """

    def __init__(self, num_layers=1, hidden_size=32, input_shape=29):
        super(MovePredict, self).__init__()
        import pickle
        with open('/mnt/aoni02/katayama/chainer/keras/embedding_metrix.pkl', "rb") as f:
            embedding_metrix = pickle.load(f)
        #with open('/mnt/aoni02/katayama/nwjc_sudachi_full_abc_w2v/embedding_metrix.pkl', "rb") as f:
        #    embedding_metrix = pickle.load(f)

        self.emb = nn.Embedding.from_pretrained(torch.Tensor(embedding_metrix))

        self.lstm_word = torch.nn.LSTM(
            input_size=300,#88,  # 入力size
            hidden_size=hidden_size,  # 出力size
            num_layers=num_layers,  # stackする数
            dropout=0.5,
            batch_first=True,  # given_data.shape = (batch , frames , input_size)
            bidirectional=False
        )

        self.lstm_sent = torch.nn.LSTM(
            input_size= hidden_size+1,  # 入力size
            hidden_size= hidden_size,  # 出力size
            num_layers=num_layers,  # stackする数
            dropout=0.5,
            batch_first=True,  # given_data.shape = (batch , frames , input_size)
            bidirectional=False
        )
        self.fc1 = nn.Linear(12, 12)
        self.fc2 = nn.Linear(input_shape, 12)
        self.fc3 = nn.Linear(300,64)
        self.fc4 = nn.Linear(hidden_size,hidden_size)
        self.fc5 = nn.Linear(hidden_size,5)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, x, xp, xp2, xi, h0 = None, c0 = None):
        x = self.emb(x)
        #x = x.unsqueeze(0)
        assert len(x.shape) == 3, print('data shape is incorrect.')
        batch_size, frames = x.shape[:2]

        _,_=self.reset_state(batch_size)

        xp = xp.view(len(xp)*frames, -1)
        xp2 = xp2.view(len(xp2)*frames, -1)
        #x = F.dropout(F.relu(self.fc3(x.view(len(x)*frames,-1))),p=0.5)
        xp = F.dropout(F.relu(self.fc1(xp)),p=0.5)
        xp2 = F.dropout(F.relu(self.fc2(xp2)),p=0.5)
        #x = x.view(-1,frames,64)
        xp = xp.view(-1,frames,12)
        xp2 = xp2.view(-1, frames, 12)
        #x = torch.cat((x,xp,xp2),dim=2)
        h, _ = self.lstm_word(x, (self.h0, self.c0))
        # h, hidden = self.lstm(x, h)
        #print(h.size())
        if h0 is None:
            h0, c0 = self.reset_state(batch_size)
            print('reset state!!')
        #h0, c0 = self.reset_state(batch_size)
        xi = xi.view(1,1,1)

        #h = torch.cat((h[:,-1:,:],xi),dim=2) #言語は最終層の出力を抽出
        #h, (h0, c0) = self.lstm_sent(h, (h0, c0))
        #h = F.dropout(F.relu(self.fc4(h[:, -1, :])), p=0.5)
        y = self.fc5(h[:,-1,:])

        return y, h0, c0

    def reset_state(self, batch_size):
        self.h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        self.c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        return self.h0, self.c0


class TargetDensePredict(nn.Module):
    """
    Targetを予測するネットワーク
    """
    def __init__(self,num_layers=1,hidden_size=128):
        super(TargetDensePredict, self).__init__()
        self.fc1 = nn.Linear(hidden_size+1, hidden_size)
        self.fc1_1 = nn.Linear(hidden_size, hidden_size)
        self.fc1_2 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)
        self.vad = VADPredict()
        self.gaze = GazeTrain()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, x_img, x_spA, x_spB, x_t):
        h_A = self.vad.get_middle(x_spA)
        h_B = self.vad.get_middle(x_spB)
        h_i = self.gaze.get_middle(x_img)
        x = torch.cat((h_A, h_B, h_i, x_t.view(-1,1)), dim=1)

        h = F.dropout(F.relu(self.fc1(x)), p=0.5)
        h = F.dropout(F.relu(self.fc1_1(h)), p=0.5)
        h = F.dropout(F.relu(self.fc1_2(h)), p=0.5)
        y = self.fc2(h)
        return y


class VADPredictLLD(nn.Module):
    """
    VAD予測するネットワーク
    """

    def __init__(self, num_layers=1, hidden_size=32):
        super(VADPredictLLD, self).__init__()
        self.lstm = torch.nn.LSTM(
            input_size=hidden_size,  # 入力size
            hidden_size=hidden_size,  # 出力size
            num_layers=1,  # stackする数
            dropout=0.5,
            batch_first=True,  # given_data.shape = (batch , frames , input_size)
            bidirectional=False
        )
        self.fc1 = nn.Linear(114, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.att_fc1 = nn.Linear(hidden_size,10)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        assert len(x.shape) == 3, print('data shape is incorrect.')
        batch_size, frames = x.shape[:2]
        x = F.dropout(F.relu(self.fc1(x.view(-1,114))), p=0.5)
        h, c = self.reset_state(batch_size)
        h, _ = self.lstm(x.view(batch_size, frames, self.hidden_size), (h, c))

        h_att = F.softmax(self.att_fc1(h[:,-1,:]),dim=1)
        h = torch.sum(h * h_att.view(-1,frames,1),1)

        #h = F.dropout(F.relu(self.fc2(h[:, -1, :])), p=0.5)
        y = self.fc2(h)

        return y

    def reset_state(self, batch_size):
        self.h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        self.c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        return self.h0, self.c0


    def get_middle(self, x):
        batch_size, frames = x.shape[:2]
        x = F.dropout(F.relu(self.fc1(x.view(-1, 114))), p=0.5)
        h, c = self.reset_state(batch_size)

        h, _ = self.lstm(x.view(batch_size, frames, self.hidden_size), (h, c))

        return h[:,-1,:]
