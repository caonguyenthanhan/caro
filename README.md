# Caro (Gomoku) AI - Python

## Cách cho người khác thử dự án

### Cách 1: Chạy từ source (cần cài Python)

- Cài Python 3.10+ (khuyến nghị 3.11+)
- Tải source (zip hoặc `git clone`)
- Chạy các lệnh dưới đây trong PowerShell tại thư mục dự án

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

### Cách 2: Đóng gói `.exe` để gửi cho người khác (không cần cài Python)

Trên máy của bạn (Windows), build bản `dist` rồi nén lại và gửi thư mục `dist\CaroAI\`.

```powershell
.\build.ps1
```

File chạy sẽ nằm tại `dist\CaroAI\CaroAI.exe`.

Nếu muốn build dạng 1 file:

```powershell
.\build.ps1 -OneFile
```

File chạy sẽ nằm tại `dist\CaroAI.exe`.

## Cài đặt (chạy từ source)

```powershell
python -m pip install -r requirements.txt
```

## Chạy ứng dụng

```powershell
python main.py
```
