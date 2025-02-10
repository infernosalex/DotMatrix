# DotMatrix

DotMatrix is a full-stack web application for QR code generation and decoding, developed as a university project. The application provides a modern, user-friendly interface for creating and reading QR codes with various customization options.

## 🚀 Features

- QR Code Generation with customizable options
- QR Code Decoding from uploaded images
- Modern, responsive web interface
- RESTful API backend
- Docker support for easy deployment
- Cross-platform desktop application (Windows, macOS, Linux)

## 🛠️ Technology Stack

### Frontend
- React (TypeScript)
- Vite for build tooling
- TailwindCSS for styling
- Framer Motion for animations
- React Router for navigation
- Axios for API communication

### Backend
- Python Flask REST API
- NumPy for image processing
- Pillow (PIL) for image manipulation
- Matplotlib for visualization

### Desktop Application
- Tauri (Rust)

## 🏃‍♂️ Running the Application

### Method 1: Using Docker (Recommended for Web Version)

The entire web application can be run using Docker Compose:

1. Make sure Docker and Docker Compose are installed on your system

2. Build and start the containers:
   ```bash
   docker-compose up --build
   ```

The web application will be available at `http://localhost:8000`

### Method 2: Desktop Application

1. Download the latest release for your operating system from the releases page
2. Install the application:
   - Windows: Run the `.msi` installer
   - macOS: Mount the `.dmg` and drag the app to Applications
   - Linux: Use the `.AppImage` or distribution-specific package

### Method 3: Without Docker (Development)

#### Backend Setup
1. Navigate to the Python backend directory:
   ```bash
   cd python
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux/MacOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the Flask server:
   ```bash
   python api.py
   ```
   The backend will be available at `http://localhost:8000`

#### Frontend Setup
1. Navigate to the web directory:
   ```bash
   cd web
   ```

2. Install Node.js dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```
   The frontend will be available at `http://localhost:5173`

#### Desktop App Development Setup
1. Install Rust and Tauri prerequisites:
   - [Rust](https://www.rust-lang.org/tools/install)
   - [System Dependencies](https://tauri.app/guides/getting-started/prerequisites)

2. Navigate to the Tauri directory:
   ```bash
   cd tauri
   ```

3. Install dependencies and run in development mode:
   ```bash
   npm install
   npm run tauri dev
   ```

## 📁 Project Structure

```
DotMatrix/
├── python/                 # Backend directory
│   ├── api.py             # Flask API endpoints
│   ├── qr_image.py        # QR code generation logic
│   ├── qr_decode.py       # QR code decoding logic
│   └── requirements.txt   # Python dependencies
├── web/                   # Frontend directory
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── tools/        # API integration
│   │   └── App.tsx       # Main application component
│   ├── package.json      # Node.js dependencies
│   └── vite.config.ts    # Vite configuration
├── tauri                 # Desktop app directory
├── Dockerfile            # Docker configuration
└── docker-compose.yml    # Docker Compose configuration
```

## 📝 License

This project is MIT licensed.

## 🎓 Academic Context

This project was developed as part of the course "Arhitectura Sistemelor de Calcul" (Computer Systems Architecture) from the Faculty of Mathematics and Computer Science, University of Bucharest. It demonstrates the implementation of QR code processing algorithms.

### 👥 Students Involved

- Andreiana Bogdan-Mihail (@bogdanws)
- Barbu David (@Dv1de29)
- Chiper Ștefan (@stefanchp)
- Scânteie Alexandru-Ioan (@infernosalex)

## 📚 References

- [QR Code Wikipedia](https://en.wikipedia.org/wiki/QR_code)
- [Project Nayuki](https://www.nayuki.io/page/creating-a-qr-code-step-by-step)
- ISO/IEC 18004:2015
