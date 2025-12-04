/**
 * API service for communicating with the backend
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface ChatRequest {
    message: string;
    current_location?: {
        city: string;
        latitude: number;
        longitude: number;
    } | null;
}

export interface ChatResponse {
    response: string;
    location?: {
        city: string;
        latitude: number;
        longitude: number;
    } | null;
    prediction?: {
        rain_mm: number;
        rain_log: number;
        chance_of_rain: number;
    } | null;
}

export const chatAPI = {
    async sendMessage(request: ChatRequest): Promise<ChatResponse> {
        const response = await fetch(`${API_BASE_URL}/api/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(request),
        });

        if (!response.ok) {
            throw new Error(`API request failed: ${response.statusText}`);
        }

        return response.json();
    },

    async healthCheck(): Promise<{ status: string }> {
        const response = await fetch(`${API_BASE_URL}/health`);

        if (!response.ok) {
            throw new Error('Health check failed');
        }

        return response.json();
    },
};
