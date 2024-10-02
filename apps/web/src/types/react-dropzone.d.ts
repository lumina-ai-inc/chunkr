declare module 'react-dropzone' {
    import { DragEvent } from 'react';

    interface DropzoneProps {
        accept?: Record<string, string[]>;
        disabled?: boolean;
        maxSize?: number;
        minSize?: number;
        multiple?: boolean;
        onDrop?: (acceptedFiles: File[], rejectedFiles: File[], event: DragEvent<HTMLElement>) => void;
        onDropAccepted?: (files: File[], event: DragEvent<HTMLElement>) => void;
        onDropRejected?: (files: File[], event: DragEvent<HTMLElement>) => void;
        onFileDialogCancel?: () => void;
        noClick?: boolean;
    }

    interface DropzoneState {
        isDragActive: boolean;
        isDragAccept: boolean;
        isDragReject: boolean;
        draggedFiles: File[];
        acceptedFiles: File[];
        rejectedFiles: File[];
    }

    export function useDropzone(options?: DropzoneProps): DropzoneState & {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        getRootProps: (props?: any) => any;
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        getInputProps: (props?: any) => any;
        open: () => void;
    };
}