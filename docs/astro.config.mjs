import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// Deployed at https://ericlbuehler.github.io/mistral.rs/
// Adjust `site` + `base` if we move to docs.mistral.rs.
export default defineConfig({
  site: 'https://ericlbuehler.github.io',
  base: '/mistral.rs',
  integrations: [
    starlight({
      title: 'mistral.rs',
      description: 'Fast, flexible LLM inference engine written in Rust.',
      social: [
        { icon: 'github', label: 'GitHub', href: 'https://github.com/EricLBuehler/mistral.rs' },
        { icon: 'discord', label: 'Discord', href: 'https://discord.gg/SZrecqK8qw' },
      ],
      editLink: {
        baseUrl: 'https://github.com/EricLBuehler/mistral.rs/edit/master/docs/',
      },
      sidebar: [
        {
          label: 'Start here',
          slug: 'start-here',
        },
        {
          label: 'Tutorials',
          autogenerate: { directory: 'tutorials' },
        },
        {
          label: 'Guides',
          autogenerate: { directory: 'guides' },
        },
        {
          label: 'Reference',
          autogenerate: { directory: 'reference' },
        },
        {
          label: 'Explanation',
          autogenerate: { directory: 'explanation' },
        },
      ],
      customCss: ['./src/styles/custom.css'],
    }),
  ],
});
