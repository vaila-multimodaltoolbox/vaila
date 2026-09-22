# Use official lightweight Node.js image
FROM node:22-slim

# Set working directory
WORKDIR /app

# Copy package files and install dependencies using lockfile
COPY package*.json ./
RUN npm ci --omit=dev

# Copy the application code
COPY index.js .

# Expose the application port
EXPOSE 8080

# Run the application as a non-root user for security.
# The official images define `node` as uid/gid 1000 — keep this in sync with
# runAsUser/runAsGroup in deployment.yaml.
USER node

# Start the application
CMD [ "node", "index.js" ]
