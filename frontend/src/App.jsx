import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import Analyze from './pages/Analyze';
import Compare from './pages/Compare';
import Explainability from './pages/Explainability';
import ModelInfo from './pages/ModelInfo';
import About from './pages/About';

function App() {
  return (
    <BrowserRouter>
      <div className="flex min-h-screen bg-[#09090b] text-white">
        <Sidebar />
        <main className="flex-1 overflow-auto bg-[#09090b]">
          <div className="p-8 max-w-7xl mx-auto">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/analyze" element={<Analyze />} />
              <Route path="/compare" element={<Compare />} />
              <Route path="/explainability" element={<Explainability />} />
              <Route path="/model-info" element={<ModelInfo />} />
              <Route path="/about" element={<About />} />
            </Routes>
          </div>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
