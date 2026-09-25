class SystemSettings:
    def __init__(self):
        # Trạng thái mặc định
        self.camera_enabled = True
        self.danger_analysis_enabled = True
        self.chatbot_enabled = True
        
    def update_settings(self, settings_dict):
        if 'camera_enabled' in settings_dict:
            self.camera_enabled = settings_dict['camera_enabled']
        if 'danger_analysis_enabled' in settings_dict:
            self.danger_analysis_enabled = settings_dict['danger_analysis_enabled']
        if 'chatbot_enabled' in settings_dict:
            self.chatbot_enabled = settings_dict['chatbot_enabled']
            
    def get_settings(self):
        return {
            'camera_enabled': self.camera_enabled,
            'danger_analysis_enabled': self.danger_analysis_enabled,
            'chatbot_enabled': self.chatbot_enabled
        }
