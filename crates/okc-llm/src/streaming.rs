//! Shared streaming helpers for SSE parsing.

use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use futures::{Stream, StreamExt};
use reqwest::Response;

/// A parsed SSE event.
#[derive(Debug, Clone)]
pub struct SseEvent {
    pub data: String,
}

/// Convert an HTTP response body into a stream of SSE events.
pub fn into_sse_stream(resp: Response) -> impl Stream<Item = anyhow::Result<SseEvent>> {
    resp.bytes_stream()
        .map(|chunk| chunk.map_err(|e| anyhow::anyhow!(e)))
        .into_sse_events()
}

/// Extension trait to turn a byte stream into SSE events.
pub trait IntoSseEvents {
    fn into_sse_events(self) -> SseStream;
}

impl<S> IntoSseEvents for S
where
    S: Stream<Item = anyhow::Result<Bytes>> + Send + 'static,
{
    fn into_sse_events(self) -> SseStream {
        SseStream {
            buffer: String::new(),
            inner: Box::pin(self),
        }
    }
}

pub struct SseStream {
    buffer: String,
    inner: Pin<Box<dyn Stream<Item = anyhow::Result<Bytes>> + Send>>,
}

impl Stream for SseStream {
    type Item = anyhow::Result<SseEvent>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            // Try to extract a complete event from the buffer
            if let Some(pos) = self.buffer.find("\n\n") {
                let chunk = self.buffer.split_off(pos + 2);
                let event_text = std::mem::replace(&mut self.buffer, chunk);
                let data = event_text
                    .lines()
                    .filter_map(|line| line.strip_prefix("data: "))
                    .collect::<Vec<_>>()
                    .join("\n");
                if !data.is_empty() {
                    return Poll::Ready(Some(Ok(SseEvent { data })));
                }
                continue;
            }

            // Need more data
            match self.inner.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(bytes))) => {
                    self.buffer.push_str(&String::from_utf8_lossy(&bytes));
                }
                Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(e))),
                Poll::Ready(None) => {
                    // EOF — drain remaining
                    if !self.buffer.is_empty() {
                        let data = self.buffer
                            .lines()
                            .filter_map(|line| line.strip_prefix("data: "))
                            .collect::<Vec<_>>()
                            .join("\n");
                        self.buffer.clear();
                        if !data.is_empty() {
                            return Poll::Ready(Some(Ok(SseEvent { data })));
                        }
                    }
                    return Poll::Ready(None);
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}
