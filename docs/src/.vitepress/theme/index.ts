// .vitepress/theme/index.ts
//
// Custom Kraken.jl theme. Extends DocumenterVitepress's default theme template
// verbatim (so the Nolebase readability menu, tabs, VersionPicker, AuthorBadge,
// JuliaMono fonts via ./style.css, and docstring styling via ./docstrings.css all
// keep working) and adds our own ./custom.css on top for the home hero backdrop
// and gradient feature-card titles.
//
// DocumenterVitepress only substitutes its default theme files for the three names
// index.ts / style.css / docstrings.css *when each is missing* from
// docs/src/.vitepress/theme/ (see vitepress_config.jl `!isfile` guards). We provide
// only index.ts + custom.css here, so the package still drops in its stock
// style.css and docstrings.css (fonts + Julia hero gradient vars we reuse), while
// this index.ts is preserved because it now exists in source.
import { h } from 'vue'
import DefaultTheme from 'vitepress/theme'
import type { Theme as ThemeConfig } from 'vitepress'

import {
  NolebaseEnhancedReadabilitiesMenu,
  NolebaseEnhancedReadabilitiesScreenMenu,
} from '@nolebase/vitepress-plugin-enhanced-readabilities/client'

import VersionPicker from "@/VersionPicker.vue"
import AuthorBadge from '@/AuthorBadge.vue'
import Authors from '@/Authors.vue'
import DownloadMenu from './components/DownloadMenu.vue'

import { enhanceAppWithTabs } from 'vitepress-plugin-tabs/client'

import '@nolebase/vitepress-plugin-enhanced-readabilities/client/style.css'
import './style.css'      // DocumenterVitepress default (JuliaMono fonts + Julia hero gradient)
import './docstrings.css' // DocumenterVitepress default (docstring blocks)
import './custom.css'     // Kraken: home hero backdrop + gradient feature titles

export const Theme: ThemeConfig = {
  extends: DefaultTheme,
  Layout() {
    return h(DefaultTheme.Layout, null, {
      'nav-bar-content-after': () => [
        h(NolebaseEnhancedReadabilitiesMenu), // Enhanced Readabilities menu
      ],
      // A enhanced readabilities menu for narrower screens (usually smaller than iPad Mini)
      'nav-screen-content-after': () => h(NolebaseEnhancedReadabilitiesScreenMenu),
    })
  },
  enhanceApp({ app, router, siteData }) {
    enhanceAppWithTabs(app);
    app.component('VersionPicker', VersionPicker);
    app.component('AuthorBadge', AuthorBadge)
    app.component('Authors', Authors)
    app.component('DownloadMenu', DownloadMenu)
  }
}
export default Theme
