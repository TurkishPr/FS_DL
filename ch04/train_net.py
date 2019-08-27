# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

#MNIST 데이터 로드
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000  #학습 iter 횟수
train_size = x_train.shape[0]
print(x_train.shape[0])
batch_size = 100 #한번의 학습 iter때 사용할 input 개수
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1) #한epoch의 크기를 설정해서 그 횟수 마다 정확도를 계산/저장/출력 하기 위함

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)  #batch_size개 random숫자들을 0~60k 에서 뽑는다
    x_batch = x_train[batch_mask] #training set에서 그 인덱스를 가져온다.
    t_batch = t_train[batch_mask] #답에서도 동일한 인덱스를 뽑아온다.
    
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch) #더 빠른 gradient 계산
    
    #Weight 업데이트.
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key] #각 weight와 bias를 방금 구한 grad*lr로 업데이트
    
    loss = network.loss(x_batch, t_batch) #x_batch, 이번 학습 batch를 가지고 forward prop을한다. 예측 결과와 답을 가지고 크로스 엔트로피 에러를 계산.
    train_loss_list.append(loss) #loss 리스트에 저장
    # print("loss = " + str(loss))
    if i % iter_per_epoch == 0: #한 epoch마다 정확도 계산/저장/출력
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

#그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()