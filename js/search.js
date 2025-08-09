;(() => {
  // Search modal functionality
  const searchModal = document.getElementById('search-modal');
  const searchModalClose = document.getElementById('search-modal-close');
  const searchModalInput = document.getElementById('search-modal-input');
  
  let pagefindUI = null;

  // Initialize PagefindUI when modal is first opened
  const initializeSearch = () => {
    if (!pagefindUI && searchModalInput) {
      pagefindUI = new PagefindUI({
        element: "#search-modal-input",
        showSubResults: true,
        showImages: true,
        processResult: (result) => {
          return result;
        }
      });
    }
  };

  // Open search modal
  window.openSearchModal = () => {
    if (searchModal) {
      initializeSearch();
      searchModal.classList.add('active');
      document.body.classList.add('search-modal-active');
      document.body.style.overflow = 'hidden';
      
      // Focus on search input after a short delay
      setTimeout(() => {
        const searchInput = searchModalInput.querySelector('input[type="search"]');
        if (searchInput) {
          searchInput.focus();
        }
      }, 100);
    }
  };

  // Close search modal
  const closeSearchModal = () => {
    if (searchModal) {
      searchModal.classList.remove('active');
      document.body.classList.remove('search-modal-active');
      document.body.style.overflow = '';
    }
  };

  // Event listeners
  if (searchModalClose) {
    searchModalClose.addEventListener('click', closeSearchModal);
  }

  // Close modal on Escape key
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && searchModal?.classList.contains('active')) {
      closeSearchModal();
    }
    
    // Open modal on Ctrl/Cmd + K
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
      e.preventDefault();
      window.openSearchModal();
    }
  });

  // Close modal when clicking outside content
  if (searchModal) {
    searchModal.addEventListener('click', (e) => {
      if (e.target === searchModal) {
        closeSearchModal();
      }
    });
  }

  // Legacy search functionality for search page
  const searchForm = document.getElementById('search-form');
  const searchBox = document.getElementById('searchbox');
  const searchResults = document.getElementById('search-results');

  if (searchForm && searchBox && searchResults) {
    let searchIndex = [];

    searchBox.select();

    const doSearch = keyword => {
      const results = [];
      const resultsEl = [];

      for (const currentItem of searchIndex) {
        if (JSON.stringify(currentItem).search(keyword) !== -1) {
          results.push(currentItem);
        }
      }

      if (results.length > 0) {
        for (const currentResult of results) {
          const currentResultEl = document.createElement('article');
          currentResultEl.classList.add('post-list-item');
          
          let categoriesHtml = '';
          if (currentResult.categories) {
            const categories = currentResult.categories.map(category => 
              `<span>${category}</span>`
            ).join('');
            categoriesHtml = `<div class="categories${
              document.body.attributes['data-uppercase-categories'].value
                ? ' text-uppercase'
                : ''
            }">${categories}</div>`;
          }
          
          currentResultEl.innerHTML = `
<a href="${currentResult.url}">
  <div class="content">
    ${categoriesHtml}
    <div class="title">${currentResult.title}</div>
  </div>
</a>
`;
          resultsEl.push(currentResultEl);
        }
      } else {
        const el = document.createElement('div');
        el.className = 'no-results';
        el.innerHTML = 'No results found.';
        resultsEl.push(el);
      }

      searchResults.innerHTML = '';
      for (const element of resultsEl) {
        searchResults.appendChild(element);
      }
    }

    searchForm.addEventListener('submit', ev => {
      ev.preventDefault();
      if (searchIndex.length > 0) {
        doSearch(searchBox.value);
      } else {
        fetch(
          document.body.attributes['data-config-root'].value +
            document.body.attributes['data-search-path'].value,
        )
          .then(res => res.json())
          .then(data => {
            searchIndex = data;
            doSearch(searchBox.value);
          });
      }
    });
  }
})()