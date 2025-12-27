#!/usr/bin/env python3
"""
Simple HTTP server for the webviewer.
Serves master_record.json files and images, and provides an API to list available cases.
"""

import os
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import mimetypes

# Configuration
# Try to detect if we're running in WSL or Windows
import platform

def detect_environment():
    """Detect if running in WSL or Windows and return appropriate paths"""
    try:
        if os.path.exists("/proc/version"):
            with open("/proc/version", "r") as f:
                if "microsoft" in f.read().lower():
                    # Running in WSL
                    BASE_DIR = "/home/qj/projects/latin_bho"
                    return (
                        os.path.join(BASE_DIR, "cp40_processing", "output"),
                        os.path.join(BASE_DIR, "input_images")
                    )
    except Exception:
        pass
    
    # Running on Windows (accessing WSL) or fallback
    return (
        r"\\wsl.localhost\Ubuntu\home\qj\projects\latin_bho\cp40_processing\output",
        r"\\wsl.localhost\Ubuntu\home\qj\projects\latin_bho\input_images"
    )

OUTPUT_DIR, INPUT_IMAGES_DIR = detect_environment()

PORT = 8000


class WebViewerHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        # API endpoint: list available cases
        if path == "/api/cases":
            self.handle_list_cases()
        # API endpoint: get master_record.json for a case
        elif path == "/api/case" and parsed_path.query:
            self.handle_get_case(parsed_path.query)
        # API endpoint: get image
        elif path.startswith("/api/image/"):
            self.handle_get_image(path)
        # Serve static files (webviewer.html)
        elif path == "/" or path == "/webviewer.html":
            self.handle_static_file("webviewer.html")
        else:
            self.send_error(404, "Not Found")
    
    def send_cors_headers(self):
        """Add CORS headers to the response"""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
    
    def send_error(self, code, message=None, explain=None):
        """Override to include CORS headers in error responses"""
        try:
            self.send_response(code, message)
            self.send_cors_headers()
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            if message:
                self.wfile.write(f"<h1>{code} {message}</h1>".encode("utf-8"))
        except Exception:
            pass  # Ignore errors when sending error response
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()
    
    def handle_list_cases(self):
        """List all available cases (subdirectories with master_record.json)"""
        try:
            cases = []
            if os.path.exists(OUTPUT_DIR):
                for item in os.listdir(OUTPUT_DIR):
                    case_dir = os.path.join(OUTPUT_DIR, item)
                    if os.path.isdir(case_dir):
                        master_record_path = os.path.join(case_dir, "master_record.json")
                        if os.path.exists(master_record_path):
                            # Try to read case metadata for display
                            try:
                                with open(master_record_path, "r", encoding="utf-8") as f:
                                    data = json.load(f)
                                    metadata = data.get("case_metadata", {})
                                    case_info = {
                                        "id": item,
                                        "group_id": metadata.get("group_id", item),
                                        "roll_number": metadata.get("roll_number", ""),
                                        "rotulus_number": metadata.get("rotulus_number", ""),
                                        "county": metadata.get("county", ""),
                                    }
                                    cases.append(case_info)
                            except Exception as e:
                                # If we can't read metadata, just use the directory name
                                cases.append({
                                    "id": item,
                                    "group_id": item,
                                    "roll_number": "",
                                    "rotulus_number": "",
                                    "county": "",
                                })
            
            cases.sort(key=lambda x: x["group_id"])
            
            self.send_response(200)
            self.send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(cases, indent=2).encode("utf-8"))
        except Exception as e:
            self.send_error(500, f"Error listing cases: {str(e)}")
    
    def handle_get_case(self, query_string):
        """Get master_record.json for a specific case"""
        try:
            params = parse_qs(query_string)
            case_id = params.get("id", [None])[0]
            
            if not case_id:
                self.send_error(400, "Missing case id parameter")
                return
            
            case_dir = os.path.join(OUTPUT_DIR, case_id)
            master_record_path = os.path.join(case_dir, "master_record.json")
            
            if not os.path.exists(master_record_path):
                self.send_error(404, f"Case {case_id} not found")
                return
            
            with open(master_record_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self.send_response(200)
            self.send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8"))
        except Exception as e:
            self.send_error(500, f"Error loading case: {str(e)}")
    
    def find_image_file(self, filename):
        """
        Find an image file with flexible matching (case-insensitive, spacing variations).
        Returns the full path to the image file, or None if not found.
        """
        from urllib.parse import unquote
        
        # URL decode the filename
        filename = unquote(filename)
        
        if not os.path.exists(INPUT_IMAGES_DIR):
            return None
        
        # Normalize filename for matching
        base_name = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1] or ".jpg"
        
        # Try exact match first
        exact_path = os.path.join(INPUT_IMAGES_DIR, filename)
        if os.path.exists(exact_path):
            return exact_path
        
        # Try case-insensitive match
        for file in os.listdir(INPUT_IMAGES_DIR):
            if file.lower() == filename.lower():
                return os.path.join(INPUT_IMAGES_DIR, file)
        
        # Try normalized name (handle spacing variations)
        normalized_base = base_name.replace(" ", "-").replace("_", "-")
        for file in os.listdir(INPUT_IMAGES_DIR):
            file_base = os.path.splitext(file)[0]
            file_ext = os.path.splitext(file)[1] or ".jpg"
            normalized_file_base = file_base.replace(" ", "-").replace("_", "-")
            
            if normalized_file_base.lower() == normalized_base.lower() and file_ext.lower() == ext.lower():
                return os.path.join(INPUT_IMAGES_DIR, file)
        
        return None
    
    def handle_get_image(self, path):
        """Serve an image file from input_images directory"""
        try:
            # Extract filename from path like /api/image/filename.jpg
            filename = path.replace("/api/image/", "")
            
            image_path = self.find_image_file(filename)
            
            if not image_path:
                self.send_error(404, f"Image {filename} not found")
                return
            
            # Determine content type
            content_type, _ = mimetypes.guess_type(image_path)
            if not content_type:
                content_type = "image/jpeg"  # Default to JPEG
            
            with open(image_path, "rb") as f:
                image_data = f.read()
            
            self.send_response(200)
            self.send_cors_headers()
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(image_data)))
            self.end_headers()
            self.wfile.write(image_data)
        except Exception as e:
            self.send_error(500, f"Error loading image: {str(e)}")
    
    def handle_static_file(self, filename):
        """Serve static HTML file"""
        try:
            file_path = os.path.join(os.path.dirname(__file__), filename)
            if not os.path.exists(file_path):
                self.send_error(404, f"File {filename} not found")
                return
            
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            self.send_response(200)
            self.send_cors_headers()
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(content.encode("utf-8"))
        except Exception as e:
            self.send_error(500, f"Error serving file: {str(e)}")
    
    def log_message(self, format, *args):
        """Log requests for debugging"""
        print(f"{self.address_string()} - {format % args}")


def main():
    print(f"Starting webviewer server on http://localhost:{PORT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Exists: {os.path.exists(OUTPUT_DIR)}")
    print(f"Input images directory: {INPUT_IMAGES_DIR}")
    print(f"Exists: {os.path.exists(INPUT_IMAGES_DIR)}")
    print(f"\nOpen http://localhost:{PORT}/webviewer.html in your browser")
    print("Press Ctrl+C to stop the server\n")
    
    try:
        server = HTTPServer(("0.0.0.0", PORT), WebViewerHandler)
        print(f"Server listening on http://0.0.0.0:{PORT} (also accessible via http://localhost:{PORT})")
        server.serve_forever()
    except OSError as e:
        if e.errno == 98 or "Address already in use" in str(e):
            print(f"\nError: Port {PORT} is already in use.")
            print("Please stop the other server or use a different port.")
        else:
            print(f"\nError starting server: {e}")
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()


if __name__ == "__main__":
    main()

