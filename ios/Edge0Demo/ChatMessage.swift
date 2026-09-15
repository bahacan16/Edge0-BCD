import Foundation

struct ChatMessage: Identifiable {
    enum Role {
        case user, assistant, system
    }

    let id = UUID()
    let role: Role
    var text: String
}
