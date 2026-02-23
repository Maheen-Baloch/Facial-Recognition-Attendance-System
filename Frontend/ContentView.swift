import SwiftUI

// 🔑 Replace this with your Ngrok public URL
let baseURL = "https://undeflected-braxton-infinite.ngrok-free.dev"

// MARK: - Model for Student
struct Student: Codable, Identifiable {
    let id = UUID()            // SwiftUI needs unique id
    let name: String
    let reg_no: String
}

struct ContentView: View {
    @State private var students: [Student] = []
    @State private var attendance: String = ""
    @State private var statusMessage: String = ""
    @State private var showCamera = false
    @State private var capturedImage: UIImage?

    
    // For Register Student
    @State private var newName: String = ""
    @State private var newRegNo: String = ""
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                
                Text("👤 Facial Recognition Attendance")
                    .font(.title2)
                    .bold()
                
                Button("1. Mark Attendance (Live)") {
                    showCamera = true
                }
                .buttonStyle(MainButtonStyle(color: .blue))
                
                Button("2. Register New Student") {
                    registerStudent()
                }
                .buttonStyle(MainButtonStyle(color: .green))
                
                Button("3. Show All Students") {
                    fetchStudents()
                }
                .buttonStyle(MainButtonStyle(color: .orange))
                
                Button("4. Show Attendance by Date") {
                    fetchAttendance()
                }
                .buttonStyle(MainButtonStyle(color: .purple))
                
                Button("5. Exit") {
                    exitApp()
                }
                .buttonStyle(MainButtonStyle(color: .red))
                
                VStack(spacing: 10) {
                    TextField("Student Name", text: $newName)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                    TextField("Registration No", text: $newRegNo)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                }
                .padding(.top)
                
                ScrollView {
                    Text(statusMessage)
                        .foregroundColor(.gray)
                        .padding()
                }
                
                Spacer()
            }
            .padding()
            .navigationTitle("Attendance App")
        }
        .sheet(isPresented: $showCamera) {
            ImagePicker(image: $capturedImage)
        }
        .onChange(of: capturedImage) { _ in
            if let img = capturedImage {
                uploadAttendance(image: img)
            }
        }
        
    }
    // MARK: - API Functions
    
    func fetchStudents() {
        guard let url = URL(string: "\(baseURL)/students") else { return }
        
        URLSession.shared.dataTask(with: url) { data, _, error in
            if let data = data {
                do {
                    let studentList = try JSONDecoder().decode([Student].self, from: data)
                    DispatchQueue.main.async {
                        self.students = studentList
                        self.statusMessage = studentList.map { "\($0.name) (\($0.reg_no))" }.joined(separator: ", ")
                    }
                } catch {
                    DispatchQueue.main.async {
                        self.statusMessage = "Failed to parse students: \(error.localizedDescription)"
                    }
                }
            } else if let error = error {
                DispatchQueue.main.async {
                    self.statusMessage = "Error: \(error.localizedDescription)"
                }
            }
        }.resume()
    }
    
    func fetchAttendance() {
        guard let url = URL(string: "\(baseURL)/attendance") else { return }
        
        URLSession.shared.dataTask(with: url) { data, _, error in
            if let data = data {
                if let attendanceText = String(data: data, encoding: .utf8) {
                    DispatchQueue.main.async {
                        self.attendance = attendanceText
                        self.statusMessage = "Attendance:\n\(attendanceText)"
                    }
                }
            } else if let error = error {
                DispatchQueue.main.async {
                    self.statusMessage = "Error: \(error.localizedDescription)"
                }
            }
        }.resume()
    }
    func uploadAttendance(image: UIImage) {
        guard let url = URL(string: "\(baseURL)/mark-attendance") else { return }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        
        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        guard let imageData = image.jpegData(compressionQuality: 0.8) else { return }
        
        var body = Data()
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"file\"; filename=\"image.jpg\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
        body.append(imageData)
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
        
        URLSession.shared.uploadTask(with: request, from: body) { data, _, error in
            if let data = data, let responseText = String(data: data, encoding: .utf8) {
                DispatchQueue.main.async {
                    statusMessage = "Attendance Result:\n\(responseText)"
                }
            } else if let error = error {
                DispatchQueue.main.async {
                    statusMessage = "Error uploading image: \(error.localizedDescription)"
                }
            }
        }.resume()
    }

    
    
    
    func registerStudent() {
        guard !newName.isEmpty, !newRegNo.isEmpty else {
            statusMessage = "Enter name and reg no first!"
            return
        }
        // Placeholder: will add image upload later
        let urlString = "\(baseURL)/register-student?name=\(newName)&reg_no=\(newRegNo)"
        guard let url = URL(string: urlString.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? "") else {
            statusMessage = "Invalid URL"
            return
        }
        
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        
        URLSession.shared.dataTask(with: request) { data, _, error in
            if let data = data, let responseText = String(data: data, encoding: .utf8) {
                DispatchQueue.main.async {
                    statusMessage = "Registered: \(responseText)"
                }
            } else if let error = error {
                DispatchQueue.main.async {
                    statusMessage = "Error: \(error.localizedDescription)"
                }
            }
        }.resume()
    }
    
    func exitApp() {
        statusMessage = "Exit tapped (iOS apps stay open)"
    }
}

// MARK: - Button Style
struct MainButtonStyle: ButtonStyle {
    var color: Color
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .frame(maxWidth: .infinity)
            .padding()
            .background(color.opacity(configuration.isPressed ? 0.5 : 0.8))
            .foregroundColor(.white)
            .cornerRadius(10)
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
