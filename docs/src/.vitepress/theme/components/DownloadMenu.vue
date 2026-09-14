<script setup lang="ts">
// DownloadMenu — a floating download dropdown for an example/tutorial page.
//
// A small download icon button floats at the TOP-RIGHT of the doc content; a
// click toggles a dropdown listing each file as a real download link
// (`<a :href download>`). Styled for the #1b1b1f dark Vitepress page.
//
// Usage (emitted as raw markdown from a Literate `.jl` source, near the title):
//   <DownloadMenu :files="[
//     {label:'poiseuille.krk', href:'/downloads/poiseuille/poiseuille.krk'},
//     {label:'poiseuille.csv', href:'/downloads/poiseuille/poiseuille.csv'},
//     {label:'poiseuille.py',  href:'/downloads/poiseuille/poiseuille.py'}]" />
//
// Files are served from docs/src/public/downloads/... (public/ -> site root).
import { ref, onMounted, onBeforeUnmount } from 'vue'

interface DownloadFile {
  label: string
  href: string
  type?: string
}

defineProps<{ files: DownloadFile[] }>()

const open = ref(false)
const root = ref<HTMLElement | null>(null)

function toggle() {
  open.value = !open.value
}

function onDocClick(e: MouseEvent) {
  if (root.value && !root.value.contains(e.target as Node)) {
    open.value = false
  }
}

function onKey(e: KeyboardEvent) {
  if (e.key === 'Escape') open.value = false
}

onMounted(() => {
  document.addEventListener('click', onDocClick)
  document.addEventListener('keydown', onKey)
})
onBeforeUnmount(() => {
  document.removeEventListener('click', onDocClick)
  document.removeEventListener('keydown', onKey)
})

// Derive a short uppercase badge (e.g. KRK, CSV, PY) from the file label.
function badge(f: DownloadFile): string {
  if (f.type) return f.type.toUpperCase()
  const dot = f.label.lastIndexOf('.')
  return dot >= 0 ? f.label.slice(dot + 1).toUpperCase() : 'FILE'
}
</script>

<template>
  <div ref="root" class="kraken-download" :class="{ open }">
    <button
      class="kraken-download__btn"
      type="button"
      aria-label="Download example files"
      :aria-expanded="open"
      @click.stop="toggle"
    >
      <!-- download icon -->
      <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
        <path
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
          d="M12 3v12m0 0l-4-4m4 4l4-4M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"
        />
      </svg>
      <span class="kraken-download__label">Files</span>
    </button>

    <transition name="kraken-download-fade">
      <ul v-if="open" class="kraken-download__menu" role="menu">
        <li v-for="f in files" :key="f.href" role="none">
          <a
            class="kraken-download__item"
            role="menuitem"
            :href="f.href"
            :download="f.label"
            @click="open = false"
          >
            <span class="kraken-download__badge">{{ badge(f) }}</span>
            <span class="kraken-download__name">{{ f.label }}</span>
          </a>
        </li>
      </ul>
    </transition>
  </div>
</template>

<style scoped>
/* Float top-right inside the VPDoc content column. The doc <main> is the
   nearest positioned ancestor in Vitepress; absolute keeps it pinned at the
   top-right of the article without disturbing the prose flow. */
.kraken-download {
  position: absolute;
  top: 12px;
  right: 0;
  z-index: 20;
  font-family: var(--vp-font-family-base);
}

.kraken-download__btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 10px;
  font-size: 13px;
  font-weight: 500;
  line-height: 1;
  color: var(--vp-c-text-2, #c9c9cf);
  background: var(--vp-c-bg-soft, #1b1b1f);
  border: 1px solid var(--vp-c-divider, #2e2e32);
  border-radius: 8px;
  cursor: pointer;
  transition: color 0.2s, border-color 0.2s, background 0.2s;
}

.kraken-download__btn:hover,
.kraken-download.open .kraken-download__btn {
  color: var(--vp-c-brand-1, #ff6b6b);
  border-color: var(--vp-c-brand-1, #ff6b6b);
  background: var(--vp-c-bg-elv, #202127);
}

.kraken-download__label {
  letter-spacing: 0.01em;
}

.kraken-download__menu {
  position: absolute;
  top: calc(100% + 6px);
  right: 0;
  min-width: 220px;
  margin: 0;
  padding: 6px;
  list-style: none;
  background: var(--vp-c-bg-elv, #202127);
  border: 1px solid var(--vp-c-divider, #2e2e32);
  border-radius: 10px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45);
}

.kraken-download__item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 10px;
  font-size: 13px;
  color: var(--vp-c-text-1, #e7e7ea);
  text-decoration: none;
  border-radius: 7px;
  transition: background 0.15s, color 0.15s;
}

.kraken-download__item:hover {
  background: var(--vp-c-default-soft, #2a2a30);
  color: var(--vp-c-brand-1, #ff6b6b);
}

.kraken-download__badge {
  flex: 0 0 auto;
  min-width: 34px;
  padding: 2px 6px;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-align: center;
  color: var(--vp-c-bg, #1b1b1f);
  background: var(--vp-c-brand-1, #ff6b6b);
  border-radius: 5px;
}

.kraken-download__name {
  font-family: var(--vp-font-family-mono);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.kraken-download-fade-enter-active,
.kraken-download-fade-leave-active {
  transition: opacity 0.15s ease, transform 0.15s ease;
}
.kraken-download-fade-enter-from,
.kraken-download-fade-leave-to {
  opacity: 0;
  transform: translateY(-4px);
}

@media (max-width: 640px) {
  /* On narrow screens, drop it into the flow so it can't overlap the title. */
  .kraken-download {
    position: static;
    margin: 0 0 12px;
  }
  .kraken-download__menu {
    right: auto;
    left: 0;
  }
}
</style>
