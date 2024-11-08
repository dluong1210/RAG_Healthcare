# Hệ thống Chatbot cho Healthcare: Ứng dụng RAG

## Giới thiệu
Đây là một hệ thống chatbot tích hợp Retrieval-Augmented Generation (RAG) để hỗ trợ tư vấn sức khỏe. Chatbot có khả năng tìm kiếm thông tin liên quan đến sức khỏe từ cơ sở dữ liệu văn bản và trả lời các câu hỏi liên quan đến bệnh lý.

## Công nghệ sử dụng
- **Langchain**: Để xây dựng các thành phần chính của hệ thống truy xuất và xử lý ngôn ngữ tự nhiên.
- **Ollama**: Để quản lý mô hình ngôn ngữ lớn (LLM) Llama cho việc tạo câu trả lời.
- **Llama**: Sử dụng mô hình ngôn ngữ Llama3.2 để tạo phản hồi cho người dùng.
- **Hugging Face**: Cung cấp embeddings để tạo vector cho văn bản và thực hiện tìm kiếm tương đồng.

## Hướng dẫn cài đặt và chạy
1. **Cài đặt các thư viện cần thiết**:
   - Tạo một môi trường ảo và kích hoạt nó (nếu cần thiết):
     ```bash
     python -m venv env
     source env/bin/activate      # Đối với Linux/macOS
     env\Scripts\activate         # Đối với Windows
     ```
   - Cài đặt các thư viện Python:
     ```bash
     pip install -r requirements.txt
     ```
2. **Chạy ứng dụng**:
   - Có thể bổ sung thêm data vào file "./data/data.txt"
   - Chạy ứng dụng bằng cách sử dụng:
     ```bash
     python app.py
     ```
   - Ứng dụng sẽ khởi động giao diện Gradio để người dùng có thể tương tác.

2. **Truy cập vào chatbot**:
   - Mở trình duyệt và truy cập vào địa chỉ mà Gradio cung cấp để bắt đầu trò chuyện với chatbot.

## Hướng dẫn sử dụng
- Đặt các câu hỏi về triệu chứng, phương pháp phòng ngừa, hoặc các bệnh lý cụ thể để nhận được tư vấn từ chatbot.
- Chatbot sẽ trả lời các câu hỏi dựa trên thông tin trong cơ sở dữ liệu và mô hình ngôn ngữ Llama. Nếu câu hỏi không liên quan đến sức khỏe, chatbot sẽ hiển thị một phản hồi chung.

Các câu hỏi ví dụ:
- "What are the symptoms of cardiovascular disease?"
- "How can I prevent diabetes?"
- "What treatments are available for cancer?"
