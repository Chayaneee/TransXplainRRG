import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

# ------ Helper Functions ------
def data2device(data, device='cpu'):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (tuple, list)):
        return type(data)(data2device(item, device) for item in data)
    elif isinstance(data, dict):
        return {k: data2device(v, device) for k, v in data.items()}
    else:
        raise TypeError('Unsupported datatype! Must be Tensor/List/Tuple/Dict.')

def data_concat(batch_data, dim=0):
    first_item = batch_data[0]
    if isinstance(first_item, torch.Tensor):
        return torch.cat(batch_data, dim=dim)
    elif isinstance(first_item, (tuple, list)):
        return type(first_item)(
            data_concat([batch[row] for batch in batch_data], dim=dim) 
            for row in range(len(first_item))
        )
    elif isinstance(first_item, dict):
        return {key: data_concat([batch[key] for batch in batch_data], dim=dim) for key in first_item}
    else:
        raise TypeError('Unsupported datatype! Must be Tensor/List/Tuple/Dict.')

def distribute_data_to_model(model, input_data):
    if isinstance(input_data, torch.Tensor):
        return model(input_data)
    elif isinstance(input_data, (tuple, list)):
        return model(*input_data)
    elif isinstance(input_data, dict):
        return model(**input_data)
    else:
        raise TypeError('Unsupported datatype! Use Tensor/List/Tuple/Dict.')

def map_args_to_kwargs(inputs, keys=None):
    if keys is not None:
        if isinstance(inputs, dict):
            return inputs
        elif isinstance(inputs, torch.Tensor):
            return {keys[0]: inputs}
        elif isinstance(inputs, (tuple, list)):
            assert len(inputs) == len(keys)
            return dict(zip(keys, inputs))
        else:
            raise TypeError('Unsupported datatype for mapping arguments.')
    return inputs

# ------ Core Functions ------
def train_one_epoch(loader, model, optimizer, loss_fn, scheduler=None, device='cpu', src_keys=None, tgt_keys=None, out_keys=None, amp_scaler=None):
    model.train()
    total_loss = 0

    progress = tqdm(loader)
    for batch_idx, (inputs, targets) in enumerate(progress):
        inputs = data2device(inputs, device)
        targets = data2device(targets, device)

        inputs = map_args_to_kwargs(inputs, src_keys)
        targets = map_args_to_kwargs(targets, tgt_keys)

        if amp_scaler:
            with torch.cuda.amp.autocast():
                predictions = distribute_data_to_model(model, inputs)
                predictions = map_args_to_kwargs(predictions, out_keys)
                loss = loss_fn(predictions, targets)

            total_loss += loss.item()
            progress.set_description(f'Loss: {total_loss / (batch_idx + 1):.4f}')

            optimizer.zero_grad()
            amp_scaler.scale(loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()
            if scheduler:
                scheduler.step()
        else:
            predictions = distribute_data_to_model(model, inputs)
            predictions = map_args_to_kwargs(predictions, out_keys)
            loss = loss_fn(predictions, targets)

            total_loss += loss.item()
            progress.set_description(f'Loss: {total_loss / (batch_idx + 1):.4f}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

    return total_loss / len(loader)

def evaluate(loader, model, loss_fn=None, device='cpu', return_preds=True, src_keys=None, tgt_keys=None, out_keys=None, selected_outputs=[]):
    model.eval()
    total_loss = 0
    all_predictions, all_targets = [], []

    with torch.no_grad():
        progress = tqdm(loader)
        for batch_idx, (inputs, targets) in enumerate(progress):
            inputs = data2device(inputs, device)
            targets = data2device(targets, device)

            inputs = map_args_to_kwargs(inputs, src_keys)
            targets = map_args_to_kwargs(targets, tgt_keys)

            predictions = distribute_data_to_model(model, inputs)
            predictions = map_args_to_kwargs(predictions, out_keys)

            if loss_fn:
                loss = loss_fn(predictions, targets)
                total_loss += loss.item()
            progress.set_description(f'Loss: {total_loss / (batch_idx + 1):.4f}')

            if return_preds:
                if selected_outputs:
                    predictions = [predictions[i] for i in selected_outputs]
                    targets = [targets[i] for i in selected_outputs]
                all_predictions.append(data2device(predictions, 'cpu'))
                all_targets.append(data2device(targets, 'cpu'))

    if return_preds:
        all_predictions = data_concat(all_predictions)
        all_targets = data_concat(all_targets)
        return total_loss / len(loader), all_predictions, all_targets
    return total_loss / len(loader)

def save_checkpoint(filepath, model, optimizer=None, scheduler=None, epoch=-1, stats=None):
    checkpoint = {
        'epoch': epoch,
        'stats': stats,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }
    torch.save(checkpoint, filepath)

# def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
#     checkpoint = torch.load(filepath)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     if optimizer:
#         try:
#             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         except Exception as e:
#             print(f"Optimizer load failed: {e}")
#     if scheduler:
#         try:
#             scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
#         except Exception as e:
#             print(f"Scheduler load failed: {e}")
#     return checkpoint.get('epoch', -1), checkpoint.get('stats', None)
    

def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(filepath)

    # --- Model Statistics ---
    epoch = checkpoint['epoch']
    stats = checkpoint['stats']

    # --- Model Parameters ---
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            # Input optimizer doesn't fit the checkpoint one --> should be ignored
            print('Cannot load the optimizer')

    if scheduler is not None:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except:
            # Input scheduler doesn't fit the checkpoint one --> should be ignored
            print('Cannot load the scheduler')

    return epoch, stats
  
