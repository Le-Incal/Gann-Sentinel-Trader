import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Trades from './pages/Trades'
import Signals from './pages/Signals'
import Debates from './pages/Debates'
import ScanCycles from './pages/ScanCycles'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="trades" element={<Trades />} />
          <Route path="signals" element={<Signals />} />
          <Route path="debates" element={<Debates />} />
          <Route path="scans" element={<ScanCycles />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default App
