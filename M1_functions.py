def train(model:nn.Module = None, loss_fn =loss_fn ,lr = 0.1,epochs = 3, trainDataLoader=trainDataLoader  , testDataLoader = testDataLoader , batch_size = 32 ):
  optimizer = torch.optim.Adam(params = model.parameters() , lr = lr)
  from tqdm.auto import tqdm

  for epoch in tqdm(range(epochs)):
    model.train()
    iterLoss = 0
    for batch , (X,y) in enumerate(trainDataLoader):
      logits = model(X)
      loss = loss_fn(logits , y)
      iterLoss += loss
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      if(batch % 400 == 0 ): print(f"seen {batch * batch_size} / {len(trainDataLoader)*batch_size}")
    overallLoss = iterLoss / len(trainDataLoader)

    model.eval()
    with torch.inference_mode():
      testLoss = 0
      testAcc = 0
      for batch , (X , y_test) in enumerate(testDataLoader):
        test_logits = model(X)
        test_loss = loss_fn(test_logits , y_test)
        testLoss += test_loss
        currAcc=hf.accuracy_fn(y_true = y_test , y_pred = test_logits.argmax(dim = 1))
        testAcc += currAcc
    overallTestLoss = testLoss / len(testDataLoader)
    overallTestAcc = testAcc / len(testDataLoader)

    print(f"epoch {epoch} | train loss : {overallLoss:3f} , test loss : {overallTestLoss:.3f} | test accuracy {overallTestAcc:3f}%")



def show_grid(row:int = 5, col:int = 5  ,test_data:torch.utils.data.datasets=test_data , Cnn:int = 0 ,  model:nn.Module):
"""
def pred(self , X):
    self.eval()
    with torch.inference_mode():
      return self(X)

"""
  fig = plt.figure(figsize = (row*2 , col*2))
  correct_count = 0
  for i in range(1, row*col +1 ):
    fig.add_subplot(row ,col , i)
    rndIdx = torch.randint(0,len(test_data) , size = [1]).item()
    img , label = test_data[rndIdx]

    if(not Cnn): pred = model.pred(img).argmax(dim=1).item()
    else: pred = model.pred(img.unsqueeze(dim=1)).argmax(dim=1).item()
    correct = (pred == label)
    correct_count+= correct

    plt.title(f"{classNames[pred]} | {classNames[label]}" , c = "green" if correct else "red" , fontsize = 8)
    plt.imshow(img.permute(1,2,0))
    plt.axis(False)
  print(f"correct : {correct_count} , Wrong : {row*col - correct_count}")
    #return {"correct" : correct_count , "Wrong" : row*col - correct_count}
