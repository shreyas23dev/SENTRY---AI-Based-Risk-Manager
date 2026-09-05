/**
 * graph.js — Sentinel D3 Force-Directed Knowledge Graph Visualizer
 * =================================================================
 * 
 * Renders the real Payment Knowledge Graph returned by the Sentinel backend.
 * Supports:
 *   - 1-hop and 2-hop dynamic traversal
 *   - Interactive Zoom, Pan, Fit, and Reset
 *   - Node category styling matching Stitch design tokens
 *   - Fraud Path Highlighting (elevating evidence paths and confirmed fraud)
 *   - Node Click -> Node Inspector Drawer population
 */

(function (window) {
  'use strict';

  // Stitch Theme Colors for Node Categories
  const NODE_COLORS = {
    target: '#00F2FE',       // Primary Cyan (Dominant)
    fraud: '#EF4444',        // Critical Crimson
    high_risk: '#F59E0B',    // Amber / Warning
    Transaction: '#00F2FE',  // Cyan
    CustomerEntity: '#10B981',// Emerald
    Entity: '#10B981',       // Emerald
    Device: '#A855F7',       // Purple / Violet
    Card: '#F59E0B',         // Amber
    Address: '#38BDF8',      // Sky Blue
    Email: '#EC4899',        // Pink
    Network: '#6366F1',      // Indigo
    default: '#89CEFF',      // Desaturated Cyan
  };

  const NODE_ICONS = {
    Transaction: 'receipt_long',
    CustomerEntity: 'person',
    Entity: 'person',
    Device: 'devices',
    Card: 'credit_card',
    Address: 'location_on',
    Email: 'mail',
    Network: 'hub',
    default: 'circle',
  };

  class KnowledgeGraphRenderer {
    constructor(containerSelector, options = {}) {
      this.container = document.querySelector(containerSelector);
      this.options = options;
      this.svg = null;
      this.g = null;
      this.simulation = null;
      this.zoomBehavior = null;
      this.currentData = null;
      this.highlightFraudPaths = false;
      this.selectedNodeId = null;
      this.maxHops = 2;
      this.onNodeSelected = options.onNodeSelected || null;
    }

    init() {
      if (!this.container) return;
      this.container.innerHTML = '';

      const width = this.container.clientWidth || 800;
      const height = this.container.clientHeight || 520;

      // Create SVG
      this.svg = d3.select(this.container)
        .append('svg')
        .attr('width', '100%')
        .attr('height', '100%')
        .attr('viewBox', `0 0 ${width} ${height}`)
        .attr('class', 'w-full h-full select-none cursor-grab active:cursor-grabbing');

      // Add defs for glow and arrow markers
      const defs = this.svg.append('defs');

      // Arrow Marker (Normal)
      defs.append('marker')
        .attr('id', 'edge-arrow')
        .attr('viewBox', '0 -5 10 10')
        .attr('refX', 22)
        .attr('refY', 0)
        .attr('markerWidth', 6)
        .attr('markerHeight', 6)
        .attr('orient', 'auto')
        .append('path')
        .attr('d', 'M0,-5L10,0L0,5')
        .attr('fill', '#3e4850');

      // Arrow Marker (Suspicious/Fraud)
      defs.append('marker')
        .attr('id', 'edge-arrow-fraud')
        .attr('viewBox', '0 -5 10 10')
        .attr('refX', 22)
        .attr('refY', 0)
        .attr('markerWidth', 7)
        .attr('markerHeight', 7)
        .attr('orient', 'auto')
        .append('path')
        .attr('d', 'M0,-5L10,0L0,5')
        .attr('fill', '#EF4444');

      // Container group with zoom behavior
      this.g = this.svg.append('g').attr('class', 'graph-viewport');

      this.zoomBehavior = d3.zoom()
        .scaleExtent([0.2, 4])
        .on('zoom', (event) => {
          this.g.attr('transform', event.transform);
        });

      this.svg.call(this.zoomBehavior);
    }

    render(data) {
      if (!this.svg) this.init();
      if (!data || !data.nodes || data.nodes.length === 0) {
        this.renderEmptyState();
        return;
      }

      this.currentData = JSON.parse(JSON.stringify(data)); // Clone data
      const width = this.container.clientWidth || 800;
      const height = this.container.clientHeight || 520;

      // Clear previous layers
      this.g.selectAll('*').remove();

      // Node map for fast lookup
      const nodeMap = new Map(this.currentData.nodes.map(n => [n.id, n]));

      // Clean links
      const links = this.currentData.edges
        .filter(e => nodeMap.has(e.source) && nodeMap.has(e.target))
        .map(e => ({ ...e }));

      // Setup D3 Force Simulation
      this.simulation = d3.forceSimulation(this.currentData.nodes)
        .force('link', d3.forceLink(links).id(d => d.id).distance(d => (d.is_suspicious ? 90 : 120)))
        .force('charge', d3.forceManyBody().strength(-400))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(d => (d.is_target ? 42 : 28)));

      // Render Edges
      const linkGroup = this.g.append('g').attr('class', 'links');
      const link = linkGroup.selectAll('line')
        .data(links)
        .enter()
        .append('line')
        .attr('stroke', d => (d.is_suspicious ? '#EF4444' : '#273647'))
        .attr('stroke-width', d => (d.is_suspicious ? 2.5 : 1.5))
        .attr('stroke-dasharray', d => (d.is_suspicious ? 'none' : '4,3'))
        .attr('marker-end', d => (d.is_suspicious ? 'url(#edge-arrow-fraud)' : 'url(#edge-arrow)'))
        .attr('opacity', 0.85);

      // Render Edge Labels
      const linkLabelGroup = this.g.append('g').attr('class', 'link-labels');
      const linkLabel = linkLabelGroup.selectAll('text')
        .data(links)
        .enter()
        .append('text')
        .attr('font-size', '9px')
        .attr('font-family', 'JetBrains Mono, monospace')
        .attr('fill', d => (d.is_suspicious ? '#EF4444' : '#88929b'))
        .attr('text-anchor', 'middle')
        .attr('dy', -4)
        .text(d => d.edge_type || d.label || '');

      // Render Nodes
      const nodeGroup = this.g.append('g').attr('class', 'nodes');
      const node = nodeGroup.selectAll('g')
        .data(this.currentData.nodes)
        .enter()
        .append('g')
        .attr('class', 'node-item cursor-pointer')
        .call(d3.drag()
          .on('start', (event, d) => {
            if (!event.active) this.simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on('drag', (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on('end', (event, d) => {
            if (!event.active) this.simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          })
        )
        .on('click', (event, d) => {
          event.stopPropagation();
          this.selectNode(d);
        });

      // Target Node Pulsing Outer Ring
      node.filter(d => d.is_target)
        .append('circle')
        .attr('r', 28)
        .attr('fill', 'none')
        .attr('stroke', '#00F2FE')
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '4,2')
        .attr('opacity', 0.6)
        .attr('class', 'animate-spin');

      // Main Node Circle
      node.append('circle')
        .attr('r', d => (d.is_target ? 22 : d.is_fraud ? 18 : 15))
        .attr('fill', d => {
          if (d.is_target) return '#051424';
          if (d.is_fraud) return '#270c10';
          return '#0d1c2d';
        })
        .attr('stroke', d => {
          if (d.is_target) return NODE_COLORS.target;
          if (d.is_fraud) return NODE_COLORS.fraud;
          if (d.is_high_risk) return NODE_COLORS.high_risk;
          return NODE_COLORS[d.node_type] || NODE_COLORS.default;
        })
        .attr('stroke-width', d => (d.is_target ? 3 : d.is_fraud ? 2.5 : 1.5))
        .attr('class', 'transition-all duration-200');

      // Node Label Text
      node.append('text')
        .attr('dy', d => (d.is_target ? 36 : 28))
        .attr('text-anchor', 'middle')
        .attr('fill', '#d4e4fa')
        .attr('font-size', d => (d.is_target ? '12px' : '10px'))
        .attr('font-weight', d => (d.is_target ? '600' : '400'))
        .attr('font-family', 'Inter, sans-serif')
        .text(d => d.label || d.id);

      // Node Category Subtitle
      node.append('text')
        .attr('dy', d => (d.is_target ? 48 : 38))
        .attr('text-anchor', 'middle')
        .attr('fill', '#88929b')
        .attr('font-size', '8px')
        .attr('font-family', 'JetBrains Mono, monospace')
        .text(d => {
          if (d.is_fraud) return 'CONFIRMED FRAUD';
          if (d.is_target) return 'TARGET TRANSACTION';
          return d.node_type ? d.node_type.toUpperCase() : '';
        });

      // Simulation Tick Handler
      this.simulation.on('tick', () => {
        link
          .attr('x1', d => d.source.x)
          .attr('y1', d => d.source.y)
          .attr('x2', d => d.target.x)
          .attr('y2', d => d.target.y);

        linkLabel
          .attr('x', d => (d.source.x + d.target.x) / 2)
          .attr('y', d => (d.source.y + d.target.y) / 2);

        node.attr('transform', d => `translate(${d.x},${d.y})`);
      });

      // Auto-fit initial view
      setTimeout(() => this.fit(), 350);
    }

    selectNode(nodeData) {
      this.selectedNodeId = nodeData.id;

      // Update visual selection borders
      this.g.selectAll('.node-item circle:nth-child(2), .node-item circle:nth-child(1)')
        .attr('filter', d => (d.id === nodeData.id ? 'drop-shadow(0 0 8px #00F2FE)' : 'none'));

      if (this.onNodeSelected) {
        this.onNodeSelected(nodeData);
      }
    }

    selectNodeById(nodeId) {
      if (!this.currentData || !this.currentData.nodes) return;
      const target = this.currentData.nodes.find(n => n.id === nodeId || String(n.id).includes(String(nodeId)));
      if (target) {
        this.selectNode(target);
      }
    }

    toggleFraudHighlight(enabled) {
      this.highlightFraudPaths = enabled;
      if (!this.g) return;

      if (enabled) {
        // Dim benign nodes and edges
        this.g.selectAll('.links line')
          .attr('opacity', d => (d.is_suspicious ? 1.0 : 0.15))
          .attr('stroke-width', d => (d.is_suspicious ? 3.5 : 1));

        this.g.selectAll('.link-labels text')
          .attr('opacity', d => (d.is_suspicious ? 1.0 : 0.1));

        this.g.selectAll('.node-item')
          .attr('opacity', d => (d.is_target || d.is_fraud || d.is_high_risk ? 1.0 : 0.25));
      } else {
        // Reset full visibility
        this.g.selectAll('.links line')
          .attr('opacity', 0.85)
          .attr('stroke-width', d => (d.is_suspicious ? 2.5 : 1.5));

        this.g.selectAll('.link-labels text')
          .attr('opacity', 1.0);

        this.g.selectAll('.node-item')
          .attr('opacity', 1.0);
      }
    }

    zoomIn() {
      if (this.svg && this.zoomBehavior) {
        this.svg.transition().duration(250).call(this.zoomBehavior.scaleBy, 1.3);
      }
    }

    zoomOut() {
      if (this.svg && this.zoomBehavior) {
        this.svg.transition().duration(250).call(this.zoomBehavior.scaleBy, 0.75);
      }
    }

    reset() {
      if (this.svg && this.zoomBehavior) {
        this.svg.transition().duration(300).call(
          this.zoomBehavior.transform,
          d3.zoomIdentity
        );
      }
    }

    fit() {
      if (!this.currentData || !this.currentData.nodes.length) return;
      const width = this.container.clientWidth || 800;
      const height = this.container.clientHeight || 520;

      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      this.currentData.nodes.forEach(n => {
        if (n.x < minX) minX = n.x;
        if (n.x > maxX) maxX = n.x;
        if (n.y < minY) minY = n.y;
        if (n.y > maxY) maxY = n.y;
      });

      if (!isFinite(minX) || !isFinite(maxX)) return;

      const graphWidth = maxX - minX + 100;
      const graphHeight = maxY - minY + 100;
      const scale = Math.min(1.5, Math.max(0.4, 0.85 / Math.max(graphWidth / width, graphHeight / height)));
      const midX = (minX + maxX) / 2;
      const midY = (minY + maxY) / 2;

      const transform = d3.zoomIdentity
        .translate(width / 2, height / 2)
        .scale(scale)
        .translate(-midX, -midY);

      this.svg.transition().duration(400).call(this.zoomBehavior.transform, transform);
    }

    renderEmptyState() {
      this.g.selectAll('*').remove();
      const width = this.container.clientWidth || 800;
      const height = this.container.clientHeight || 520;

      this.g.append('text')
        .attr('x', width / 2)
        .attr('y', height / 2)
        .attr('text-anchor', 'middle')
        .attr('fill', '#88929b')
        .attr('font-size', '14px')
        .attr('font-family', 'Inter, sans-serif')
        .text('No graph context available for this transaction.');
    }
  }

  window.KnowledgeGraphRenderer = KnowledgeGraphRenderer;
})(window);
