import { NavLink } from 'react-router-dom';
import {
  Activity,
  ScanLine,
  GitCompare,
  Info,
  BookOpen,
  Layers,
  Stethoscope,
} from 'lucide-react';

const navItems = [
  { to: '/', label: 'Dashboard', icon: Activity, exact: true },
  { to: '/analyze', label: 'Analyze', icon: ScanLine },
  { to: '/compare', label: 'Compare', icon: GitCompare },
  { to: '/explainability', label: 'Explainability', icon: Layers },
  { to: '/model-info', label: 'Model Info', icon: Info },
  { to: '/about', label: 'About', icon: BookOpen },
];

export default function Sidebar() {
  return (
    <aside className="w-64 shrink-0 bg-[#101114] border-r border-[#202228] h-screen sticky top-0 flex flex-col select-none">
      {/* Brand Header */}
      <div className="px-5 py-5 border-b border-[#202228]">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-[#1c1e24] border border-[#2e313b] flex items-center justify-center text-white shadow-sm">
            <Stethoscope className="w-4 h-4 text-white" />
          </div>
          <div>
            <div className="text-sm font-bold text-white tracking-tight leading-tight">ChestVision AI</div>
            <div className="text-xs text-[#8e929f] leading-tight mt-0.5">X-ray Diagnostic AI</div>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-4 space-y-1">
        {navItems.map(({ to, label, icon: Icon, exact }) => (
          <NavLink
            key={to}
            to={to}
            end={exact}
            className={({ isActive }) =>
              `nav-link ${isActive ? 'active' : ''}`
            }
          >
            <Icon className="w-4 h-4 shrink-0" />
            <span className="font-medium text-sm">{label}</span>
          </NavLink>
        ))}
      </nav>

      {/* Disclaimer Footer */}
      <div className="px-5 py-4 border-t border-[#202228] bg-[#0c0d10]">
        <p className="text-[11px] text-[#858997] leading-relaxed">
          Research prototype — for educational analysis, not clinical diagnosis.
        </p>
      </div>
    </aside>
  );
}
