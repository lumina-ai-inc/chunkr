self.onmessage = (evt: MessageEvent<File>) => {
  const file = evt.data;
  // FileReaderSync is available in Web Worker
  const reader = new FileReaderSync();
  const dataUrl = reader.readAsDataURL(file); // e.g. "data:…;base64,AAAA…"
  const base64 = dataUrl.split(",", 2)[1]; // strip prefix
  self.postMessage(base64);
};
