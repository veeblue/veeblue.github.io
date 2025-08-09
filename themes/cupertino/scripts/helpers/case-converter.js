/**
 * Helper functions to convert categories and tags case, and standardize cover image fields
 */

hexo.extend.helper.register('uppercase_category', function(category) {
  return category.name ? category.name.toUpperCase() : category;
});

hexo.extend.helper.register('lowercase_tag', function(tag) {
  return tag.name ? tag.name.toLowerCase() : tag;
});

hexo.extend.helper.register('process_categories', function(categories) {
  if (!categories || !categories.length) return categories;
  
  return categories.map(category => {
    const processed = Object.assign({}, category);
    processed.name = category.name ? category.name.toUpperCase() : category.name;
    return processed;
  });
});

hexo.extend.helper.register('process_tags', function(tags) {
  if (!tags || !tags.length) return tags;
  
  return tags.map(tag => {
    const processed = Object.assign({}, tag);
    processed.name = tag.name ? tag.name.toLowerCase() : tag.name;
    return processed;
  });
});

hexo.extend.helper.register('get_cover_image', function(page) {
  // Standardize cover image field - prefer 'cover' over 'cover_image'
  return page.cover || page.cover_image;
});

hexo.extend.helper.register('get_cover_image_alt', function(page) {
  // Standardize cover image alt field - prefer 'cover_alt' over 'cover_image_alt'
  return page.cover_alt || page.cover_image_alt || '';
});

hexo.extend.helper.register('has_cover_image', function(page) {
  // Check if page has cover image using either field name
  return !!(page.cover || page.cover_image);
});