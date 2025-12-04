import { Chat } from '@/types/chat';
import { Cloud, CloudRain, Droplets } from 'lucide-react';
import { useEffect, useRef } from 'react';
import { ChatInput } from './ChatInput';
import { ChatMessage } from './ChatMessage';

interface ChatAreaProps {
  chat: Chat | null;
  isLoading: boolean;
  onSendMessage: (message: string) => void;
}

export const ChatArea = ({ chat, isLoading, onSendMessage }: ChatAreaProps) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chat?.messages]);

  return (
    <div className="flex-1 flex flex-col h-full bg-background">
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-3xl mx-auto">
          {!chat || chat.messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full min-h-[60vh] text-center animate-fade-in">
              <div className="relative mb-8">
                <CloudRain className="w-20 h-20 text-primary animate-float" />
                <Droplets className="w-8 h-8 text-primary/50 absolute -bottom-2 -right-2 animate-float-delayed" />
                <Cloud className="w-10 h-10 text-primary/40 absolute -top-2 -left-4 animate-float-delayed" />
              </div>
              <h2 className="text-2xl font-semibold text-foreground mb-2">
                Rainfall Predictor
              </h2>
              <p className="text-muted-foreground max-w-md">
                Ask me about rainfall predictions for major cities around the world.
              </p>
              <div className="mt-8 grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-md">
                {[
                  "San Francisco",
                  "Delhi",
                  "Tokyo",
                  "London",
                ].map((suggestion) => (
                  <button
                    key={suggestion}
                    onClick={() => onSendMessage(suggestion)}
                    className="px-6 py-3 text-sm text-left bg-card border border-border rounded-xl hover:bg-accent transition-colors shadow-soft"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              {chat.messages.map((message) => (
                <ChatMessage key={message.id} message={message} />
              ))}
              {isLoading && (
                <div className="flex gap-3 animate-fade-in">
                  <div className="w-9 h-9 rounded-full bg-accent flex items-center justify-center">
                    <CloudRain className="w-5 h-5 text-accent-foreground" />
                  </div>
                  <div className="bg-card border border-border rounded-2xl rounded-tl-sm px-4 py-3 shadow-soft">
                    <div className="flex gap-1">
                      <span className="w-2 h-2 bg-primary/50 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                      <span className="w-2 h-2 bg-primary/50 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                      <span className="w-2 h-2 bg-primary/50 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>
      </div>

      {/* Input Area */}
      <ChatInput onSend={onSendMessage} isLoading={isLoading} />
    </div>
  );
};
