FROM node:20-alpine

WORKDIR /app

# Cache dependencies
COPY package*.json ./
RUN npm install

# Copy source files
COPY . .

# Expose Vite default port
EXPOSE 5173

# Start Vite dev server bound to all interfaces
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
