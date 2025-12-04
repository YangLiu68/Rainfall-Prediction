import { cn } from '@/lib/utils';
import { Message } from '@/types/chat';
import { format } from 'date-fns';
import { CloudRain, User } from 'lucide-react';

interface ChatMessageProps {
  message: Message;
}

export const ChatMessage = ({ message }: ChatMessageProps) => {
  const isUser = message.role === 'user';

  // Parse markdown bold syntax (**text**)
  const formatContent = (text: string) => {
    const parts = text.split(/(\*\*.*?\*\*)/g);
    return parts.map((part, index) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return <strong key={index}>{part.slice(2, -2)}</strong>;
      }
      return part;
    });
  };

  return (
    <div
      className={cn(
        'flex gap-3 animate-slide-up',
        isUser ? 'flex-row-reverse' : 'flex-row'
      )}
    >
      {/* Avatar */}
      <div
        className={cn(
          'flex-shrink-0 w-9 h-9 rounded-full flex items-center justify-center',
          isUser
            ? 'bg-primary text-primary-foreground'
            : 'bg-accent text-accent-foreground'
        )}
      >
        {isUser ? (
          <User className="w-5 h-5" />
        ) : (
          <CloudRain className="w-5 h-5" />
        )}
      </div>

      {/* Message Bubble */}
      <div
        className={cn(
          'max-w-[70%] rounded-2xl px-4 py-3 shadow-soft',
          isUser
            ? 'bg-primary text-primary-foreground rounded-tr-sm'
            : 'bg-card text-card-foreground rounded-tl-sm border border-border'
        )}
      >
        <p className="text-sm leading-relaxed whitespace-pre-wrap">
          {formatContent(message.content)}
        </p>
        <p
          className={cn(
            'text-xs mt-2',
            isUser ? 'text-primary-foreground/70' : 'text-muted-foreground'
          )}
        >
          {format(message.timestamp, 'h:mm a')}
        </p>
      </div>
    </div>
  );
};
