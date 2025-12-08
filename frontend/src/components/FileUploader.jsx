import { useState, useRef } from 'react'

export default function FileUploader({ onUpload, files }) {
    const [isDragging, setIsDragging] = useState(false)
    const fileInputRef = useRef(null)

    const handleDragOver = (e) => {
        e.preventDefault()
        setIsDragging(true)
    }

    const handleDragLeave = () => {
        setIsDragging(false)
    }

    const handleDrop = async (e) => {
        e.preventDefault()
        setIsDragging(false)
        const droppedFiles = Array.from(e.dataTransfer.files)
        await uploadFiles(droppedFiles)
    }

    const handleFileSelect = async (e) => {
        const selectedFiles = Array.from(e.target.files)
        await uploadFiles(selectedFiles)
    }

    const uploadFiles = async (fileList) => {
        // In a real app, you would upload to the backend here
        // For now, we simulate the upload and update local state

        for (const file of fileList) {
            const formData = new FormData()
            formData.append('file', file)

            try {
                const response = await fetch('http://localhost:8000/api/upload', {
                    method: 'POST',
                    body: formData,
                })
                if (!response.ok) throw new Error('Upload failed')
            } catch (error) {
                console.error('Error uploading file:', error)
            }
        }

        onUpload(fileList.map(f => ({ name: f.name, size: f.size })))
    }

    return (
        <div className="flex flex-col h-full">
            <div
                className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${isDragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'
                    }`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
            >
                <input
                    type="file"
                    ref={fileInputRef}
                    className="hidden"
                    multiple
                    onChange={handleFileSelect}
                />
                <p className="text-gray-600">
                    Drag & drop files here, or click to select
                </p>
            </div>

            <div className="mt-6 flex-1 overflow-y-auto">
                <h3 className="font-semibold text-gray-700 mb-2">Uploaded Files</h3>
                <ul className="space-y-2">
                    {files.map((file, index) => (
                        <li key={index} className="bg-gray-50 p-2 rounded text-sm flex justify-between items-center">
                            <span className="truncate">{file.name}</span>
                            <span className="text-gray-400 text-xs">{(file.size / 1024).toFixed(1)} KB</span>
                        </li>
                    ))}
                </ul>
            </div>
        </div>
    )
}
