;(() => {
  // Enhanced language detection function
  function detectLanguage(block) {
    let language = 'code';
    
    // Try to detect language from class (most reliable)
    const languageMatch = block.className.match(/language-(\w+)/);
    if (languageMatch) {
      language = languageMatch[1];
    } else {
      // Try to detect from Hexo's highlight class (e.g., "highlight json")
      const highlightMatch = block.className.match(/highlight\s+([a-zA-Z+#-]+)/);
      if (highlightMatch) {
        language = highlightMatch[1];
      } else {
        // Try to detect from other class attributes
        const classMatch = block.className.match(/(?:^|\s)([a-zA-Z+#-]+)(?=\s|$)/);
        if (classMatch && !['highlight', 'hljs', 'figure'].includes(classMatch[1])) {
          language = classMatch[1];
        } else {
          // Try to detect from data attributes
          const dataLang = block.getAttribute('data-lang');
          const dataLanguage = block.getAttribute('data-language');
          if (dataLang) {
            language = dataLang;
          } else if (dataLanguage) {
            language = dataLanguage;
          }
        }
      }
    }
    
    // Normalize language names
    const languageMap = {
      'js': 'javascript',
      'ts': 'typescript',
      'jsx': 'jsx',
      'tsx': 'tsx',
      'py': 'python',
      'python': 'python',
      'rb': 'ruby',
      'go': 'go',
      'rs': 'rust',
      'c': 'c',
      'cpp': 'cpp',
      'c++': 'cpp',
      'cs': 'csharp',
      'java': 'java',
      'kt': 'kotlin',
      'scala': 'scala',
      'swift': 'swift',
      'objc': 'objective-c',
      'php': 'php',
      'html': 'html',
      'css': 'css',
      'scss': 'scss',
      'sass': 'sass',
      'less': 'less',
      'sql': 'sql',
      'bash': 'bash',
      'sh': 'shell',
      'zsh': 'zsh',
      'fish': 'fish',
      'powershell': 'powershell',
      'ps1': 'powershell',
      'yaml': 'yaml',
      'yml': 'yaml',
      'json': 'json',
      'xml': 'xml',
      'toml': 'toml',
      'ini': 'ini',
      'dockerfile': 'dockerfile',
      'docker': 'dockerfile',
      'nginx': 'nginx',
      'apache': 'apache',
      'vim': 'vim',
      'lua': 'lua',
      'r': 'r',
      'matlab': 'matlab',
      'perl': 'perl',
      'julia': 'julia',
      'haskell': 'haskell',
      'elm': 'elm',
      'dart': 'dart',
      'flutter': 'dart',
      'rust': 'rust',
      'solidity': 'solidity',
      'vue': 'vue',
      'svelte': 'svelte',
      'angular': 'typescript',
      'react': 'jsx',
      'md': 'markdown',
      'markdown': 'markdown',
      'tex': 'latex',
      'latex': 'latex',
      'diff': 'diff',
      'patch': 'diff',
      'git': 'git',
      'mermaid': 'mermaid',
      'plantuml': 'plantuml',
      'graphql': 'graphql',
      'protobuf': 'protobuf',
      'thrift': 'thrift',
      'avro': 'avro',
      'groovy': 'groovy',
      'clojure': 'clojure',
      'elixir': 'elixir',
      'erlang': 'erlang',
      'fsharp': 'fsharp',
      'ocaml': 'ocaml',
      'reason': 'reason',
      'pascal': 'pascal',
      'fortran': 'fortran',
      'cobol': 'cobol',
      'assembly': 'assembly',
      'asm': 'assembly',
      'nasm': 'assembly',
      'makefile': 'makefile',
      'cmake': 'cmake',
      'gradle': 'gradle',
      'maven': 'maven',
      'npm': 'npm',
      'yarn': 'yarn',
      'pip': 'pip',
      'conda': 'conda',
      'gem': 'gem',
      'cargo': 'cargo',
      'go-mod': 'go',
      'rust-mod': 'rust'
    };
    
    // Clean up language name
    language = language.toLowerCase().replace(/[^a-z0-9+#-]/g, '');
    language = languageMap[language] || language;
    
    // Special handling for common patterns
    if (language === 'node' || language === 'nodejs') {
      return 'javascript';
    } else if (language === 'ts' || language === 'tsx') {
      return 'typescript';
    } else if (language === 'js' || language === 'jsx' || language === 'mjs') {
      return 'javascript';
    } else if (language === 'py' || language === 'py3' || language === 'py2' || language === 'python') {
      return 'python';
    } else if (language === 'rb') {
      return 'ruby';
    } else if (language === 'go') {
      return 'go';
    } else if (language === 'rs') {
      return 'rust';
    } else if (language === 'c') {
      return 'c';
    } else if (language === 'cpp' || language === 'cc' || language === 'cxx' || language === 'c++') {
      return 'cpp';
    } else if (language === 'cs') {
      return 'csharp';
    } else if (language === 'kt') {
      return 'kotlin';
    } else if (language === 'swift') {
      return 'swift';
    } else if (language === 'php') {
      return 'php';
    } else if (language === 'html' || language === 'htm') {
      return 'html';
    } else if (language === 'css') {
      return 'css';
    } else if (language === 'scss' || language === 'sass') {
      return language;
    } else if (language === 'sql') {
      return 'sql';
    } else if (language === 'sh' || language === 'shell' || language === 'bash' || language === 'zsh' || language === 'fish') {
      return 'bash';
    } else if (language === 'ps1' || language === 'powershell') {
      return 'powershell';
    } else if (language === 'yaml' || language === 'yml') {
      return 'yaml';
    } else if (language === 'json') {
      return 'json';
    } else if (language === 'xml') {
      return 'xml';
    } else if (language === 'toml') {
      return 'toml';
    } else if (language === 'ini') {
      return 'ini';
    } else if (language === 'dockerfile' || language === 'docker') {
      return 'dockerfile';
    } else if (language === 'nginx') {
      return 'nginx';
    } else if (language === 'vim') {
      return 'vim';
    } else if (language === 'lua') {
      return 'lua';
    } else if (language === 'r') {
      return 'r';
    } else if (language === 'perl') {
      return 'perl';
    } else if (language === 'julia') {
      return 'julia';
    } else if (language === 'haskell') {
      return 'haskell';
    } else if (language === 'dart') {
      return 'dart';
    } else if (language === 'rust') {
      return 'rust';
    } else if (language === 'vue') {
      return 'vue';
    } else if (language === 'svelte') {
      return 'svelte';
    } else if (language === 'md' || language === 'markdown') {
      return 'markdown';
    } else if (language === 'tex' || language === 'latex') {
      return 'latex';
    } else if (language === 'diff' || language === 'patch') {
      return 'diff';
    } else if (language === 'git') {
      return 'git';
    } else if (language === 'mermaid') {
      return 'mermaid';
    } else if (language === 'graphql') {
      return 'graphql';
    } else if (language === 'protobuf') {
      return 'protobuf';
    } else if (language === 'makefile') {
      return 'makefile';
    } else if (language === 'cmake') {
      return 'cmake';
    } else if (language === 'gradle') {
      return 'gradle';
    } else if (language === 'maven') {
      return 'maven';
    }
    
    return language;
  }

  // Function to wrap code blocks with macOS-style UI
  function wrapCodeBlocks() {
    const content = document.querySelector('.content');
    if (!content) return;

    const codeBlocks = content.querySelectorAll('figure.highlight, .highlight');
    
    codeBlocks.forEach(block => {
      // Skip if already wrapped
      if (block.closest('.code-block-wrapper')) return;

      // Create wrapper
      const wrapper = document.createElement('div');
      wrapper.className = 'code-block-wrapper';

      // Create header
      const header = document.createElement('div');
      header.className = 'code-block-header';

      // Create traffic lights
      const trafficLights = document.createElement('div');
      trafficLights.className = 'traffic-lights';
      
      const closeLight = document.createElement('div');
      closeLight.className = 'traffic-light close';
      
      const minimizeLight = document.createElement('div');
      minimizeLight.className = 'traffic-light minimize';
      
      const maximizeLight = document.createElement('div');
      maximizeLight.className = 'traffic-light maximize';
      
      trafficLights.appendChild(closeLight);
      trafficLights.appendChild(minimizeLight);
      trafficLights.appendChild(maximizeLight);

      // Enhanced language detection
      const languageDisplay = document.createElement('div');
      languageDisplay.className = 'code-language';
      
      let language = detectLanguage(block);
      // Capitalize first letter for better display
      languageDisplay.textContent = language.charAt(0).toUpperCase() + language.slice(1);

      // Create copy button
      const copyButton = document.createElement('button');
      copyButton.className = 'copy-button';
      copyButton.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="m4 16c-1.1 0-2-.9-2-2v-10c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>';
      copyButton.title = 'Copy code';

      // Add copy functionality
      copyButton.addEventListener('click', async () => {
        try {
          let codeText = '';
          
          // Handle Hexo's table structure: figure.highlight > table > tr > td.code > pre > span.line
          const codeLines = block.querySelectorAll('.code .line');
          if (codeLines.length > 0) {
            // Extract text from each line, preserving line breaks
            codeLines.forEach((line, index) => {
              codeText += line.textContent + (index < codeLines.length - 1 ? '\n' : '');
            });
          } else {
            // Fallback: try to find code element
            const code = block.querySelector('code');
            if (!code) return;
            codeText = code.innerText || code.textContent;
          }

          // Mobile-friendly clipboard handling
          if (navigator.clipboard && window.isSecureContext) {
            // Modern browsers with secure context
            await navigator.clipboard.writeText(codeText);
          } else {
            // Fallback for mobile and older browsers
            const textArea = document.createElement('textarea');
            textArea.value = codeText;
            textArea.style.position = 'fixed';
            textArea.style.left = '-999999px';
            textArea.style.top = '-999999px';
            textArea.style.fontSize = '16px'; // Prevent zoom on iOS
            document.body.appendChild(textArea);
            textArea.focus();
            textArea.select();
            
            // Execute copy command
            const successful = document.execCommand('copy');
            document.body.removeChild(textArea);
            
            if (!successful) {
              throw new Error('Copy command failed');
            }
          }

          copyButton.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';
          copyButton.classList.add('copied');
          
          setTimeout(() => {
            copyButton.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="m4 16c-1.1 0-2-.9-2-2v-10c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>';
            copyButton.classList.remove('copied');
          }, 2000);
        } catch (err) {
          console.error('Failed to copy code:', err);
          copyButton.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';
          setTimeout(() => {
            copyButton.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="m4 16c-1.1 0-2-.9-2-2v-10c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>';
          }, 2000);
        }
      });

      // Assemble header
      header.appendChild(trafficLights);
      header.appendChild(languageDisplay);
      header.appendChild(copyButton);

      // Wrap the code block
      block.parentNode.insertBefore(wrapper, block);
      wrapper.appendChild(header);
      wrapper.appendChild(block);
      
    });
  }

  // Run on page load
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', wrapCodeBlocks);
  } else {
    wrapCodeBlocks();
  }

  // Run after content changes (for dynamic content)
  const observer = new MutationObserver(() => {
    wrapCodeBlocks();
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true
  });
})();