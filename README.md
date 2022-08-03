# Pytorch_CNN

## Pytorch로 CNN모델을 만들어 이미지 분류하기


#### 데이터 셋에 대한 평균과 표준편차 구하기
```python3

data_dir = "../input/-----------------" #데이터 셋 경로
train_ds_transform = transforms.Compose([transforms.ToTensor()]) #데이터를 오직 텐서화만 진행할 예정
train_ds = datasets.ImageFolder(data_dir+'/train', transform = train_ds_transform) #데이터 셋 텐서화 진행


meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x,_ in train_ds] 
stdRGB = [np.std(x.numpy(), axis=(1,2)) for x,_ in train_ds]



meanR = np.mean([m[0] for m in meanRGB])
#0번째 index에는 R에 대한 값이 들어있고 R의 평균값을 구한다.
meanG = np.mean([m[1] for m in meanRGB])
#1번째 index에는 G에 대한 값이 들어있고 R의 평균값을 구한다.
meanB = np.mean([m[2] for m in meanRGB])
#2번째 index에는 B에 대한 값이 들어있고 R의 평균값을 구한다.

stdR = np.mean([s[0] for s in stdRGB])
stdG = np.mean([s[1] for s in stdRGB])
stdB = np.mean([s[2] for s in stdRGB])

```

#### data augmentation

```python3
data_dir = "../input/-----------------"

#변수명에 A를 붙였던 이유는 후에 앙상블 기법을 활용하는데 모델은 그대로두고 데이터만 변형해서 데이터 증강효과를 더 보기위한 구분자이다.

#train_data에 대한 transform
#여러 transform들을 활용하기 위해 Compose를 사용한다.
train_transform_A = transforms.Compose([
                                      transforms.RandomCrop(244), #랜덤한 위치에서 244*244만큼 자른다.
                                      transforms.RandomHorizontalFlip(p=0.5), #주어진 확률값으로 수평으로 뒤집는다. 현재 50%
                                      transforms.RandomVerticalFlip(p=0.5), #주어진 확률값으로 수직으로 뒤집는다. 현재 50%
                                      transforms.ToTensor(), #데이터 텐서화.
                                      transforms.Normalize([meanR, meanG, meanB], [stdR, stdG, stdB])]) #텐서 이미지 정규화.
                                      #input[channel] = (input[channel] - mean[channel]) / std[channel]공식에 의해 RGB가 normalize된다.

#valid_data에 대한 transform
valid_transform_A = transforms.Compose([
                                      transforms.CenterCrop(244), #가운데에서 244*244만큼 자른다
                                      transforms.ToTensor(),#데이터 텐서화.
                                      transforms.Normalize([meanR, meanG, meanB ], [stdR, stdG, stdB])])#텐서 이미지 정규화.


train_data_A = datasets.ImageFolder(data_dir+'/train', transform = train_transform_A)
valid_data_A = datasets.ImageFolder(data_dir+'/train', transform = valid_transform_A)



val_size = 0.2 #valid_data의 크기를 20%로 설정
num_train = len(train_data_A) 
num_train = int(num_train) 
indices = list(range(num_train)) #길이만큼 index부여하면서 list생성한다. indices = [0 ~ int(num_train)-1]
np.random.shuffle(indices) #indices를 랜덤으로 섞는다.
split = int(np.floor(val_size * num_train)) # np.floor은 내림함수이다. 8:2로 나누기 위해 구간을 정한다.
train_idx, valid_idx = indices[split:], indices[:split] #train_dix는 20~99, valide_idx는 0~19만큼 가진다.

#SubsetRandomSampler 주어진 인덱스 목록에서 요소를 무작위로 샘플링
#dataset의 일정 부분을 나눠서 사용할 수 있다.
train_sampler_A = SubsetRandomSampler(train_idx)
valid_sampler_A = SubsetRandomSampler(valid_idx)

train_loader_A = torch.utils.data.DataLoader(train_data_A, batch_size=16, sampler = train_sampler_A)
valid_loader_A = torch.utils.data.DataLoader(valid_data_A, batch_size=16, sampler = valid_sampler_A)

```





#### model architecture
```python3
class GNN(nn.Module):
    def __init__(self):
        super().__init__()
        #input  3*244*244
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride = 1, padding=1)
        #input채널이 3, output채널을 64로 설정.  kernel_size=3으로 설정. 필터(kerndel) 이동거리는 1(stride=1)
        #padding=1로 원래 채널의 테두리에 한칸이 늘어나며 값은 0으로 들어간다.
        self.conv1_bn = nn.BatchNorm2d(64)
        #2d배치 정규화로 output 채널값을 num_features에 받는다.
        #더 쉽게 train가능하게 해주며 gradient의 흐름을 향상시킨다.
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride = 1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride = 1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride = 1, padding=1)
        self.conv4_bn = nn.BatchNorm2d(512)
        
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, stride = 1, padding=1)
        self.conv5_bn = nn.BatchNorm2d(1024)
        
        self.conv6 = nn.Conv2d(1024, 2048, kernel_size=3, stride = 1, padding=1)
        self.conv6_bn = nn.BatchNorm2d(2048)
        
        
        
        self.pool = nn.MaxPool2d((2,2) , stride=2) 
        #kernel_size = (2,2)로 해당 필터 안에 최대 값을 가져온다.
        #stride =2로 채널의 이동거리이다. MaxPool2d를 이와 같이 설정하면 한 층이 지나면 출력값이 1/2로 줄어든다.
        
        self.fc1 = nn.Linear(2048*3*3, 512)
        self.fc1_bn = nn.BatchNorm1d(512)
        #1d배치 정규화로 output값을 num_feature에 받는다.
        
        self.fc2 = nn.Linear(512, 15)
        
        self.dropout = nn.Dropout(0.5) 
        #신경망의 과적합을 줄이기 위한 대표적인 정규화 기술. 뉴런의 일부를 0으로대치하여 계산에서 제외한다.
        #0.2~0.5가 통상 
        
        self.relu = nn.ReLU(True)
        #max(0,x)를 의미하는 함수로 0보다 작으면 0이 되는 특징을 가지고 있다.
        #0이하의 입력에 대해 0을 출력함으로써 부분적으로 활성화가 가능하며 선형함수이므로 미분 계산이 매우 편리하다.
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(self.relu(self.conv2_bn(self.conv2(x))))

        x = self.pool(self.relu(self.conv3_bn(self.conv3(x))))
        x = self.pool(self.relu(self.conv4_bn(self.conv4(x))))
        
        x = self.pool(self.relu(self.conv5_bn(self.conv5(x))))
        x = self.pool(self.relu(self.conv6_bn(self.conv6(x))))
        #컨볼루션 -> 배치 정규화 -> activation func -> pooling
        
        x = x.view(-1,2048*3*3)
        #데이터 flatten

        x = self.relu(self.fc1_bn(self.fc1(x)))
        x = self.dropout(x)
        
        x = self.fc2(x)
        
        return x

```

#### class 불균형에 인해 직접적으로 가중치 설정
```python3
normedWeights1 = [0.99, 0.7, 0.7, 0.4, 0.97, 
                  0.85, 0.99, 0.7, 0.5, 0.6, 
                  0.8, 0.97, 0.7, 0.99, 0.8]
normedWeights2 = torch.FloatTensor(normedWeights1).to(device)

criterion_A = nn.CrossEntropyLoss(normedWeights2)
```


#### optimizer & scheduler

```python3
optimizer_A = optim.Adam(model_A.parameters(), lr = 1e-4, weight_decay=1e-5)
scheduler_A = optim.lr_scheduler.MultiStepLR(optimizer_A, milestones=[15,30,45], gamma=0.1, last_epoch=- 1, verbose=True)

#model_A의 파라미터를 받으며 learning rate = 1e-4, weight_decay = 1e-5로 설정했다.
#MultiStepLR learnig rate를 감소시킬 epoch을 정해준다. 감소량은 lr= gamma*lr 

```


#### model pre-trained
```python3
n_epochs = 60 #에폭 수 60

valid_loss_min = np.Inf #valid_loss를 무한으로 주고 내려갈때 마다 저장한다.

#에폭 수 만큼 0으로 채운 텐서 만들기. 나중에 그래프 그릴 때 사용한다.
train_loss = torch.zeros(n_epochs) 
valid_loss = torch.zeros(n_epochs) 

train_acc = torch.zeros(n_epochs) 
valid_acc = torch.zeros(n_epochs) 


for e in range(0, n_epochs):
        model_A.train() #model_A학습 시작
        for data, labels in train_loader_A:
            data, labels = data.to(device), labels.to(device)
            #data, labels를 cuda로 이동

            optimizer_A.zero_grad()
            #파라미터를 업데이트 한 후 새로운 batch에 대한 새 gradient를 계산해야 하기 때문에 옵티마이저를 초기화한다.
            
            logits = model_A(data) #data를 model_A에 넣어 결과 예측(계산)
            
            loss = criterion_A(logits, labels)#batch loss 계산
            
            loss.backward() #전체 파라미터에 대한 gradient를 구한다.
            optimizer_A.step()  #gradient를 각 파라미터에 적용한다.

            train_loss[e] += loss.item() #loss를 에폭 위치에 넣어준다.

            ps = F.softmax(logits, dim=1) #logits값에 softmax를 취해 0과 1사이의 값으로 바뀐다.
            top_p, top_class = ps.topk(1, dim=1) #주어진 input 텐서의 가장 큰 값 1개를 반환한다.
            equals = top_class == labels.reshape(top_class.shape) #labels의 shape을 top_class의 shape으로 reshape한 후, top_class와 비교한다.
            train_acc[e] += torch.mean(equals.type(torch.float)).item() #True, False를 float형 토치로 변환 후 평균을 에폭 index에 넣어준다.

        train_loss[e] /= len(train_loader_A) #batch_size로 정해진 train_loader_A길이(크기)로 나눈다.
        train_acc[e] /= len(train_loader_A)

        with torch.no_grad():
            model_A.eval() #validation 과정에서 사용하지 않을 layer들을 off(dropout 등)
            for data, labels in valid_loader_A:
                data, labels = data.to(device), labels.to(device)

                logits = model_A(data) #data를 model_A에 넣어 결과 예측(계산)
                loss = criterion_A(logits, labels) #batch loss 계산
                valid_loss[e] += loss.item() #loss를 에폭 위치에 넣어준다.

                ps = F.softmax(logits, dim=1) #logits값에 softmax를 취해 0과 1사이의 값으로 바뀐다.
                top_p, top_class = ps.topk(1, dim=1) #주어진 input 텐서의 가장 큰 값 1개를 반환한다.
                equals = top_class == labels.reshape(top_class.shape) #labels의 shape을 top_class의 shape으로 reshape한 후, top_class와 비교한다.
                valid_acc[e] += torch.mean(equals.type(torch.float)).item() #True, False를 float형 토치로 변환 후 평균을 에폭 index에 넣어준다.

        valid_loss[e] /= len(valid_loader_A) #batch_size로 정해진 valid_loader_A길이(크기)로 나눈다.
        valid_acc[e] /= len(valid_loader_A)
        scheduler_A.step()# 옵티마이저 매개 변수 업데이트(최적화) 단계
        
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            e, train_loss[e], valid_loss[e]))
        #train_loss와 valid_loss를 epoch마다 출력
        
        print('Epoch: {} \tTraining accuracy: {:.6f} \tValidation accuracy: {:.6f}'.format(
            e, train_acc[e], valid_acc[e]))
        #train_acc와 valid_acc를 epoch마다 출력
        
        #loss가 감소하면 state_dict저장. early stopping
        if valid_loss_min > valid_loss[e]:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss[e]))
            validpoint_A = {'model_state_dict': model_A.state_dict(),
                  'optimizer_state_dict': optimizer_A.state_dict()}
            #딕셔너리 타입으로 저장
            torch.save(validpoint_A, 'validpoint_A.pth')

            valid_loss_min = valid_loss[e] #valid_loss 갱신

```
#### loss와 accuracy 그래프로 확인

 <img width="50%" src="https://user-images.githubusercontent.com/70587454/182602366-a9fa4ac6-76e8-4993-898f-05695934365a.png"/>


#### model test 

```python3
#앙상블 soft-voting 기법 사용
for data in test_DataLoader:
    data = data.to(device)
    logits_A = model_A1(data)
    logits_B = model_A2(data)
    logits_C = model_C(data)
    logits = (logits_A+logits_B+logits_C)/3
    ps = F.softmax(logits, dim=1)
    _, top_class = ps.topk(1, dim=1)
    for i in range(len(data)):
        pred.append(top_class[i].item())
```

####  save csv file 
```python3
name=[]                                     
category =[]                                
for i in range(test_data.__len__()):        
    name.append(test_set[i].split('/')[-1]) 
    category.append(classes[pred[i]])       
    
    
tm = localtime()
csv_file_name = f"{tm.tm_mday}day_{tm.tm_hour}h_{tm.tm_min}m_{tm.tm_sec}s"


pd.DataFrame({'Id':name,'Category':category}).to_csv(f'predict_{"ff"}.csv',index=False)   

```














