import { chatAPI } from '@/services/api';
import { Chat, Message } from '@/types/chat';
import { useCallback, useEffect, useState } from 'react';

const STORAGE_KEY = 'rainfall-predictor-chats';
const LOCATION_KEY = 'rainfall-predictor-location';

const generateId = () => Math.random().toString(36).substring(2, 15);

interface StoredLocation {
  city: string;
  latitude: number;
  longitude: number;
}

const getStoredChats = (): Chat[] => {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      const chats = JSON.parse(stored);
      return chats.map((chat: Chat) => ({
        ...chat,
        createdAt: new Date(chat.createdAt),
        updatedAt: new Date(chat.updatedAt),
        messages: chat.messages.map((msg: Message) => ({
          ...msg,
          timestamp: new Date(msg.timestamp),
        })),
      }));
    }
  } catch (error) {
    console.error('Error loading chats:', error);
  }
  return [];
};

const saveChats = (chats: Chat[]) => {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(chats));
  } catch (error) {
    console.error('Error saving chats:', error);
  }
};

const getStoredLocation = (): StoredLocation | null => {
  try {
    const stored = localStorage.getItem(LOCATION_KEY);
    if (stored) {
      return JSON.parse(stored);
    }
  } catch (error) {
    console.error('Error loading location:', error);
  }
  return null;
};

const saveLocation = (location: StoredLocation | null) => {
  try {
    if (location) {
      localStorage.setItem(LOCATION_KEY, JSON.stringify(location));
    } else {
      localStorage.removeItem(LOCATION_KEY);
    }
  } catch (error) {
    console.error('Error saving location:', error);
  }
};

export const useChat = () => {
  const [chats, setChats] = useState<Chat[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [currentLocation, setCurrentLocation] = useState<StoredLocation | null>(null);

  useEffect(() => {
    const storedChats = getStoredChats();
    const storedLocation = getStoredLocation();
    setChats(storedChats);
    setCurrentLocation(storedLocation);
    if (storedChats.length > 0) {
      setActiveChatId(storedChats[0].id);
    }
  }, []);

  useEffect(() => {
    if (chats.length > 0) {
      saveChats(chats);
    }
  }, [chats]);

  useEffect(() => {
    saveLocation(currentLocation);
  }, [currentLocation]);

  const activeChat = chats.find((chat) => chat.id === activeChatId) || null;

  const createNewChat = useCallback(() => {
    const newChat: Chat = {
      id: generateId(),
      name: 'New Chat',
      messages: [],
      createdAt: new Date(),
      updatedAt: new Date(),
    };
    setChats((prev) => [newChat, ...prev]);
    setActiveChatId(newChat.id);
    return newChat.id;
  }, []);

  const sendMessage = useCallback(async (content: string) => {
    if (!content.trim()) return;

    let chatId = activeChatId;

    if (!chatId) {
      chatId = createNewChat();
    }

    const userMessage: Message = {
      id: generateId(),
      role: 'user',
      content: content.trim(),
      timestamp: new Date(),
    };

    setChats((prev) =>
      prev.map((chat) => {
        if (chat.id === chatId) {
          const isFirstMessage = chat.messages.length === 0;
          return {
            ...chat,
            name: isFirstMessage ? content.slice(0, 30) + (content.length > 30 ? '...' : '') : chat.name,
            messages: [...chat.messages, userMessage],
            updatedAt: new Date(),
          };
        }
        return chat;
      })
    );

    setIsLoading(true);

    try {
      // Call the real API
      const response = await chatAPI.sendMessage({
        message: content.trim(),
        current_location: currentLocation,
      });

      // Update location if returned
      if (response.location) {
        setCurrentLocation(response.location);
      }

      const assistantMessage: Message = {
        id: generateId(),
        role: 'assistant',
        content: response.response,
        timestamp: new Date(),
      };

      setChats((prev) =>
        prev.map((chat) => {
          if (chat.id === chatId) {
            return {
              ...chat,
              messages: [...chat.messages, assistantMessage],
              updatedAt: new Date(),
            };
          }
          return chat;
        })
      );
    } catch (error) {
      console.error('Error sending message:', error);

      // Show error message to user
      const errorMessage: Message = {
        id: generateId(),
        role: 'assistant',
        content: `Sorry, I encountered an error: ${error instanceof Error ? error.message : 'Unknown error'}. Please make sure the backend server is running.`,
        timestamp: new Date(),
      };

      setChats((prev) =>
        prev.map((chat) => {
          if (chat.id === chatId) {
            return {
              ...chat,
              messages: [...chat.messages, errorMessage],
              updatedAt: new Date(),
            };
          }
          return chat;
        })
      );
    } finally {
      setIsLoading(false);
    }
  }, [activeChatId, createNewChat, currentLocation]);

  const deleteChat = useCallback((chatId: string) => {
    setChats((prev) => {
      const filtered = prev.filter((chat) => chat.id !== chatId);
      if (activeChatId === chatId) {
        setActiveChatId(filtered.length > 0 ? filtered[0].id : null);
      }
      if (filtered.length === 0) {
        localStorage.removeItem(STORAGE_KEY);
      }
      return filtered;
    });
  }, [activeChatId]);

  const clearAllChats = useCallback(() => {
    setChats([]);
    setActiveChatId(null);
    localStorage.removeItem(STORAGE_KEY);
  }, []);

  return {
    chats,
    activeChat,
    activeChatId,
    isLoading,
    currentLocation,
    setActiveChatId,
    createNewChat,
    sendMessage,
    deleteChat,
    clearAllChats,
  };
};
