import torch
import time



def train_and_eval(model,train_iter, valid_iter, optimizer, loss_fn =nn.NLLLoss() , epochs=20,checkpoint_file_final='final.pt'):
    #TRAIN!!!!
    for epoch_idx in range(epochs):
        total_loss, num_correct = 0, 0
        total_samples = 0
        start_time = time.time()

        for train_batch in train_iter:
            X, y = train_batch.text, train_batch.label

            # Forward pass
            y_pred_log_proba = model(X)

            # Backward pass
            optimizer.zero_grad()
            loss = loss_fn(y_pred_log_proba, y)
            loss.backward()

            # Weight updates
            optimizer.step()

            # Calculate accuracy
            total_loss += loss.item()
            y_pred = torch.argmax(y_pred_log_proba, dim=1)
            num_correct += torch.sum(y_pred == y).float().item()
            total_samples+= len(train_batch) 

                
        print(f"Epoch #{epoch_idx}, loss={total_loss /(train_iter.shape[0]):.3f}, accuracy={num_correct /(total_samples):.3f}, elapsed={time.time()-start_time:.1f} sec")

    
"""
def train_wrapper(model, train_iter, valid_iter, checkpoint_file_final='final.pt'):
    train_loss_arr, train_acc_arr, val_loss_arr, val_acc_arr = [], [], [], []
    loss_min = float('inf')
    best_acc = -float('inf')

    for epoch in range(10):
        save_checkpoint = False
        train_loss, train_acc = train_model(model, train_iter, epoch)
        train_loss_arr.append(train_loss)
        train_acc_arr.append(train_acc)
        val_loss, val_acc = eval_model(model, valid_iter)
        val_loss_arr.append(val_loss)
        val_acc_arr.append(val_acc)
        
        print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')
      
        if train_loss < loss_min:
            loss_min = train_loss               
            save_checkpoint = True
            if best_acc < train_acc:
                best_acc = train_acc
                
        if save_checkpoint and checkpoint_file_final is not None:
            saved_state = dict(
                best_acc=best_acc,
                model_state=model.state_dict(),
            )
            torch.save(saved_state, checkpoint_file_final)
            print(
                f"*** Saved checkpoint {checkpoint_file_final} " f"at epoch {epoch+1}"
            )

    return train_loss_arr, train_acc_arr, val_loss_arr, val_acc_arr

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)

def train_model(model, train_iter, epoch, checkpoints: str = None):

    total_epoch_loss = 0
    total_epoch_acc = 0
    model.cuda()
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    steps = 0
    model.train()
    for idx, batch in enumerate(train_iter):
        text = batch.text
        target = batch.label
        target = torch.autograd.Variable(target).long()
        if torch.cuda.is_available():
            text = text.cuda()
            target = target.cuda()
        if (text.size()[0] != 32):
            continue
        optim.zero_grad()
        prediction = model(text)
        loss = loss_fn(prediction, target)
        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects/len(batch)
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1
            
        if steps % 100 == 0:
            print (f'Epoch: {epoch+1}, Idx: {idx+1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item(): .2f}%')

            
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()
   

    return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter)


def eval_model(model, val_iter):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            text = batch.text[0]
            if (text.size()[0] != 32):
                continue
            target = batch.label
            target = torch.autograd.Variable(target).long()
            if torch.cuda.is_available():
                text = text.cuda()
                target = target.cuda()
            prediction = model(text)
            loss = loss_fn(prediction, target)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects/len(batch)
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter)
"""















def train(model, optimizer, loss_fn, dataloader, max_epochs=100, max_batches=200, batch_size=4):
    for epoch_idx in range(max_epochs):
        total_loss, num_correct = 0, 0
        start_time = time.time()

        for batch_idx, batch in enumerate(dataloader):
            X, y = batch.text, batch.label

            # Forward pass
            y_pred_log_proba = model(X)

            # Backward pass
            optimizer.zero_grad()
            loss = loss_fn(y_pred_log_proba, y)
            loss.backward()

            # Weight updates
            optimizer.step()

            # Calculate accuracy
            total_loss += loss.item()
            y_pred = torch.argmax(y_pred_log_proba, dim=1)
            num_correct += torch.sum(y_pred == y).float().item()

            if batch_idx == max_batches-1:
                break
                
        print(f"Epoch #{epoch_idx}, loss={total_loss /(max_batches):.3f}, accuracy={num_correct /(max_batches*batch_size):.3f}, elapsed={time.time()-start_time:.1f} sec")
        

def evaluate_model(model,loss_fn, dataloader, max_epochs=100, max_batches=200, batch_size=4):
    total_loss, num_correct = 0, 0
    start_time = time.time()
    model.eval()
    for batch_idx, batch in enumerate(dataloader):
        X, y = batch.text, batch.label
        
        with torch.no_grad():
            y_pred_log_proba = model(X)
            loss = loss_fn(y_pred_log_proba, y)
        total_loss += loss.item()
        y_pred = torch.argmax(y_pred_log_proba, dim=1)
        num_correct += torch.sum(y_pred == y).float().item()
        if batch_idx == max_batches-1:
            break
    print(f" loss={total_loss /(max_batches):.3f}, accuracy={num_correct /(max_batches*batch_size):.3f}, elapsed={time.time()-start_time:.1f} sec")
                
    
    
    model.train()
        
        
def calculate_accuracy(model, dataloader, device):
    model.eval() # put in evaluation mode
    total_correct = 0
    total_images = 0
    confusion_matrix = np.zeros([10,10], int)
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            for i, l in enumerate(labels):
                confusion_matrix[l.item(), predicted[i].item()] += 1 

    model_accuracy = total_correct / total_images * 100
    return model_accuracy, confusion_matrix