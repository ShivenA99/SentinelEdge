import { useRef, useState, useCallback, useEffect } from 'react'

interface UseWebSocketOptions {
  url: string
  onMessage: (data: any) => void
  onOpen?: () => void
  onClose?: () => void
  autoConnect?: boolean
}

interface UseWebSocketReturn {
  connect: () => void
  disconnect: () => void
  send: (data: any) => void
  isConnected: boolean
  error: string | null
}

export function useWebSocket(options: UseWebSocketOptions): UseWebSocketReturn {
  const { url, onMessage, onOpen, onClose, autoConnect = false } = options
  const [isConnected, setIsConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const onMessageRef = useRef(onMessage)
  const onOpenRef = useRef(onOpen)
  const onCloseRef = useRef(onClose)

  // Keep callback refs fresh without causing reconnections
  useEffect(() => {
    onMessageRef.current = onMessage
    onOpenRef.current = onOpen
    onCloseRef.current = onClose
  }, [onMessage, onOpen, onClose])

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }
    if (wsRef.current) {
      wsRef.current.close(1000, 'Client disconnect')
      wsRef.current = null
    }
    setIsConnected(false)
  }, [])

  const connect = useCallback(() => {
    if (!url) return
    disconnect()
    setError(null)

    try {
      const ws = new WebSocket(url)
      wsRef.current = ws

      ws.onopen = () => {
        setIsConnected(true)
        setError(null)
        onOpenRef.current?.()
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          onMessageRef.current(data)
        } catch {
          // If not JSON, pass raw data
          onMessageRef.current(event.data)
        }
      }

      ws.onerror = () => {
        setError('WebSocket connection error')
      }

      ws.onclose = (event) => {
        setIsConnected(false)
        wsRef.current = null
        onCloseRef.current?.()

        // Auto-reconnect on abnormal closure (not manual disconnect)
        if (event.code !== 1000 && event.code !== 1005) {
          reconnectTimeoutRef.current = setTimeout(() => {
            connect()
          }, 3000)
        }
      }
    } catch (err) {
      setError(`Failed to connect: ${err}`)
      setIsConnected(false)
    }
  }, [url, disconnect])

  const send = useCallback((data: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(typeof data === 'string' ? data : JSON.stringify(data))
    }
  }, [])

  // Auto-connect on mount if configured
  useEffect(() => {
    if (autoConnect && url) {
      connect()
    }
    return () => {
      disconnect()
    }
  }, [autoConnect, url, connect, disconnect])

  return { connect, disconnect, send, isConnected, error }
}
