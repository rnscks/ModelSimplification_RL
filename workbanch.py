import torch
USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('CUDA 사용 가능 여부 :', USE_CUDA)
print('현재 사용 device :', device)
print('CUDA Index :', torch.cuda.current_device())
print('GPU 이름 :', torch.cuda.get_device_name())
print('GPU 개수 :', torch.cuda.device_count())
