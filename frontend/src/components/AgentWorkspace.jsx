import { useState, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'

const AGENTS = [
    { id: 'policy', name: 'Policy Tool' },
    { id: 'regulation', name: 'Regulation Tool' },
    { id: 'report', name: 'Report Tool' },
    { id: 'risk', name: 'Risk Tool' },
    { id: 'custom', name: 'Custom Agent' },
]

export default function AgentWorkspace() {
    // Initialize state from localStorage if available
    const [activeTab, setActiveTab] = useState(() => localStorage.getItem('esg_active_tab') || 'regulation')
    const [agentOutput, setAgentOutput] = useState(() => localStorage.getItem('esg_agent_output') || '')
    const [isLoading, setIsLoading] = useState(false)

    // Persist state changes
    useEffect(() => {
        localStorage.setItem('esg_active_tab', activeTab)
    }, [activeTab])

    useEffect(() => {
        localStorage.setItem('esg_agent_output', agentOutput)
    }, [agentOutput])

    const handleRunAgent = async () => {
        setIsLoading(true)
        setAgentOutput('')
        try {
            const response = await fetch(`http://localhost:8000/api/agent/${activeTab}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: 'Start analysis' }),
            })
            const data = await response.json()
            setAgentOutput(data.result || 'No output returned.')
        } catch (error) {
            setAgentOutput(`Error: ${error.message}`)
        } finally {
            setIsLoading(false)
        }
    }

    return (
        <div className="flex flex-col h-full">
            {/* Tabs */}
            <div className="flex border-b border-gray-200 bg-white">
                {AGENTS.map((agent) => (
                    <button
                        key={agent.id}
                        onClick={() => setActiveTab(agent.id)}
                        className={`px-4 py-3 text-sm font-medium focus:outline-none ${activeTab === agent.id
                            ? 'border-b-2 border-blue-500 text-blue-600'
                            : 'text-gray-500 hover:text-gray-700'
                            }`}
                    >
                        {agent.name}
                    </button>
                ))}
            </div>

            {/* Content Area */}
            <div className="flex-1 p-6 overflow-y-auto">
                <div className="mb-4 flex justify-between items-center">
                    <h2 className="text-2xl font-bold text-gray-800">
                        {AGENTS.find(a => a.id === activeTab)?.name}
                    </h2>
                    <button
                        onClick={handleRunAgent}
                        disabled={isLoading}
                        className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 disabled:opacity-50"
                    >
                        {isLoading ? 'Running...' : 'Run Agent'}
                    </button>
                </div>

                <div className="bg-white p-6 rounded-lg shadow-sm min-h-[400px]">
                    {isLoading ? (
                        <div className="flex justify-center items-center h-full text-gray-400">
                            Processing...
                        </div>
                    ) : agentOutput ? (
                        <div className="prose max-w-none">
                            <ReactMarkdown
                                components={{
                                    a: ({ node, ...props }) => (
                                        <a {...props} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline" />
                                    )
                                }}
                            >
                                {agentOutput}
                            </ReactMarkdown>
                        </div>
                    ) : (
                        <div className="text-center text-gray-400 mt-20">
                            Select an agent and click "Run Agent" to start.
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}
