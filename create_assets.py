#!/usr/bin/env python3
"""
Create placeholder assets for the GitHub page
"""

import os
from pathlib import Path

def create_assets_directory():
    """Create assets directory and placeholder files"""
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)
    
    # Create placeholder SVG files
    placeholders = {
        "jigyasa-banner.svg": """<svg width="800" height="200" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#4F46E5;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#7C3AED;stop-opacity:1" />
    </linearGradient>
  </defs>
  <rect width="800" height="200" fill="url(#grad1)" rx="10"/>
  <text x="400" y="80" font-family="Arial, sans-serif" font-size="48" font-weight="bold" fill="white" text-anchor="middle">JIGYASA</text>
  <text x="400" y="120" font-family="Arial, sans-serif" font-size="20" fill="white" text-anchor="middle">Autonomous General Intelligence System</text>
  <text x="400" y="150" font-family="Arial, sans-serif" font-size="16" fill="white" text-anchor="middle" opacity="0.8">Powered by Llama 3.1</text>
</svg>""",
        
        "architecture-diagram.svg": """<svg width="700" height="400" xmlns="http://www.w3.org/2000/svg">
  <rect width="700" height="400" fill="#f8f9fa" stroke="#e9ecef" stroke-width="2" rx="5"/>
  
  <!-- Core AGI -->
  <rect x="250" y="50" width="200" height="60" fill="#4F46E5" rx="5"/>
  <text x="350" y="85" font-family="Arial" font-size="18" fill="white" text-anchor="middle">Core AGI System</text>
  
  <!-- Components -->
  <rect x="50" y="150" width="150" height="50" fill="#10B981" rx="5"/>
  <text x="125" y="180" font-family="Arial" font-size="14" fill="white" text-anchor="middle">Ollama Wrapper</text>
  
  <rect x="225" y="150" width="150" height="50" fill="#3B82F6" rx="5"/>
  <text x="300" y="180" font-family="Arial" font-size="14" fill="white" text-anchor="middle">Performance</text>
  
  <rect x="400" y="150" width="150" height="50" fill="#8B5CF6" rx="5"/>
  <text x="475" y="180" font-family="Arial" font-size="14" fill="white" text-anchor="middle">Learning</text>
  
  <rect x="575" y="150" width="150" height="50" fill="#EC4899" rx="5"/>
  <text x="650" y="180" font-family="Arial" font-size="14" fill="white" text-anchor="middle">Autonomous</text>
  
  <!-- Connections -->
  <line x1="350" y1="110" x2="125" y2="150" stroke="#6B7280" stroke-width="2"/>
  <line x1="350" y1="110" x2="300" y2="150" stroke="#6B7280" stroke-width="2"/>
  <line x1="350" y1="110" x2="475" y2="150" stroke="#6B7280" stroke-width="2"/>
  <line x1="350" y1="110" x2="650" y2="150" stroke="#6B7280" stroke-width="2"/>
  
  <!-- Database -->
  <ellipse cx="350" cy="300" rx="80" ry="40" fill="#F59E0B"/>
  <text x="350" y="305" font-family="Arial" font-size="14" fill="white" text-anchor="middle">SQLite DB</text>
  
  <!-- Connection to DB -->
  <line x1="350" y1="200" x2="350" y2="260" stroke="#6B7280" stroke-width="2"/>
</svg>""",
        
        "performance-chart.svg": """<svg width="700" height="300" xmlns="http://www.w3.org/2000/svg">
  <rect width="700" height="300" fill="#ffffff" stroke="#e5e7eb" stroke-width="1"/>
  
  <!-- Title -->
  <text x="350" y="30" font-family="Arial" font-size="18" font-weight="bold" text-anchor="middle">Performance Improvements</text>
  
  <!-- Y-axis -->
  <line x1="50" y1="50" x2="50" y2="250" stroke="#374151" stroke-width="2"/>
  <text x="20" y="55" font-family="Arial" font-size="12" fill="#6B7280">100%</text>
  <text x="20" y="150" font-family="Arial" font-size="12" fill="#6B7280">50%</text>
  <text x="20" y="245" font-family="Arial" font-size="12" fill="#6B7280">0%</text>
  
  <!-- X-axis -->
  <line x1="50" y1="250" x2="650" y2="250" stroke="#374151" stroke-width="2"/>
  
  <!-- Bars -->
  <rect x="100" y="130" width="80" height="120" fill="#10B981" opacity="0.8"/>
  <text x="140" y="270" font-family="Arial" font-size="12" text-anchor="middle">Loops</text>
  <text x="140" y="120" font-family="Arial" font-size="14" font-weight="bold" text-anchor="middle">65%</text>
  
  <rect x="250" y="100" width="80" height="150" fill="#3B82F6" opacity="0.8"/>
  <text x="290" y="270" font-family="Arial" font-size="12" text-anchor="middle">Algorithm</text>
  <text x="290" y="90" font-family="Arial" font-size="14" font-weight="bold" text-anchor="middle">73%</text>
  
  <rect x="400" y="160" width="80" height="90" fill="#8B5CF6" opacity="0.8"/>
  <text x="440" y="270" font-family="Arial" font-size="12" text-anchor="middle">Memory</text>
  <text x="440" y="150" font-family="Arial" font-size="14" font-weight="bold" text-anchor="middle">45%</text>
  
  <rect x="550" y="180" width="80" height="70" fill="#EC4899" opacity="0.8"/>
  <text x="590" y="270" font-family="Arial" font-size="12" text-anchor="middle">Strings</text>
  <text x="590" y="170" font-family="Arial" font-size="14" font-weight="bold" text-anchor="middle">35%</text>
</svg>"""
    }
    
    # Create SVG files
    for filename, content in placeholders.items():
        filepath = assets_dir / filename
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"✅ Created {filepath}")
        
        # Also create PNG version placeholder
        if filename.endswith('.svg'):
            png_name = filename.replace('.svg', '.png')
            readme_content = f"# {png_name}\n\nThis is a placeholder for the {png_name} image.\nReplace this with an actual screenshot or graphic."
            with open(assets_dir / f"{png_name}.README", 'w') as f:
                f.write(readme_content)
    
    # Create additional placeholder files
    additional_files = {
        "demo.gif.README": "Replace this with an animated GIF showing JIGYASA in action",
        "terminal-demo.png.README": "Replace this with a screenshot of the terminal running JIGYASA"
    }
    
    for filename, content in additional_files.items():
        with open(assets_dir / filename, 'w') as f:
            f.write(f"# {filename}\n\n{content}")
        print(f"✅ Created {assets_dir / filename}")

if __name__ == "__main__":
    create_assets_directory()
    print("\n📁 Assets directory created with placeholder files")
    print("Replace these with actual images/screenshots for the best effect")